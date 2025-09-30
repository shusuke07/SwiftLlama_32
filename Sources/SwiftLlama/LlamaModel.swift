import Foundation
import llama
import os

class LlamaModel {
    private let model: Model
    private let configuration: Configuration
    private let context: OpaquePointer
    private let sampler: UnsafeMutablePointer<llama_sampler>
    private var batch: Batch
    private var tokens: [Token]
    private var temporaryInvalidCChars: [CChar] = []
    private var generatedTokenAccount: Int32 = 0
    private var ended = false
    private let n_len: Int32 = 1024
    private let logger = Logger(subsystem: "SwiftLlama", category: "LlamaModel")

    var shouldContinue: Bool {
        generatedTokenAccount < configuration.maxTokenCount && !ended
    }

    init(path: String, configuration: Configuration = .init()) throws {
        self.configuration = configuration
        llama_backend_init()
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED)
        var model_params = llama_model_default_params()
        #if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        #endif
        // モデル読み込み
        guard let loadedModel = llama_load_model_from_file(path, model_params) else {
            // 初期化失敗時でもバックエンドを解放
            llama_backend_free()
            throw SwiftLlamaError.others("Cannot load model at path \(path)")
        }
        // コンテキスト作成
        guard let createdContext = llama_new_context_with_model(loadedModel, configuration.contextParameters) else {
            llama_free_model(loadedModel)
            llama_backend_free()
            throw SwiftLlamaError.others("Cannot load model context")
        }
        // 事前チェック（self 代入前に失敗時の解放を徹底）
        let n_ctx = llama_n_ctx(createdContext)
        let n_ctx_train = llama_n_ctx_train(loadedModel)
        if n_ctx > n_ctx_train {
            llama_free(createdContext)
            llama_free_model(loadedModel)
            llama_backend_free()
            throw SwiftLlamaError.others("Model was trained on \(n_ctx_train) context but tokens \(n_ctx) specified")
        }
        // ここから self に反映
        self.model = loadedModel
        self.context = createdContext
        self.tokens = []
        self.batch = llama_batch_init(Int32(configuration.batchSize * max(1, configuration.historyLimit) * 2), 0, 1)
        self.sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(configuration.temperature))
        llama_sampler_chain_add(sampler, llama_sampler_init_softmax())
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234))
    }

    private func checkContextLength(context: Context, model: Model) throws {
        let n_ctx = llama_n_ctx(context)
        let n_ctx_train = llama_n_ctx_train(model)
        if n_ctx > n_ctx_train {
            throw SwiftLlamaError.others("Model was trained on \(n_ctx_train) context but tokens \(n_ctx) specified")
        }
    }

    func start(for prompt: Prompt) throws {
        ended = false
        tokens = tokenize(text: prompt.prompt, addBos: true)
        temporaryInvalidCChars = []
        batch.clear()
        if configuration.debugLogTokens {
            let tokenPieces: [String] = tokens.map { t in
                let bytes = tokenToCChars(token: t)
                return String(validating: bytes + [0], as: UTF8.self) ?? ""
            }
            logger.info("[SwiftLlama][init tokens] count=\(self.tokens.count)")
            for (i, t) in tokens.enumerated() {
                let piece = i < tokenPieces.count ? tokenPieces[i] : ""
                logger.debug("  [\(i)] id=\(t) piece=\(piece)")
            }
        }

        tokens.enumerated().forEach { index, token in
            batch.add(token: token, position: Int32(index), seqIDs: [0], logit: false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        generatedTokenAccount = batch.n_tokens
    }

    func `continue`() throws -> String {
        let newToken =  llama_sampler_sample(sampler, context, batch.n_tokens - 1)
        if configuration.debugLogTokens {
            let bytes = tokenToCChars(token: newToken)
            let piece = String(validating: bytes + [0], as: UTF8.self) ?? ""
            logger.debug("[SwiftLlama][gen token] id=\(newToken) piece=\(piece)")
        }

        if llama_vocab_is_eog(llama_model_get_vocab(model), newToken) || generatedTokenAccount == n_len {
            temporaryInvalidCChars.removeAll()
            ended = true
            return ""
        }


        let newTokenCChars = tokenToCChars(token: newToken)
        // NUL(0x00) を生成直後に除去してから蓄積
        let sanitizedCChars = newTokenCChars.filter { $0 != 0 }
        temporaryInvalidCChars.append(contentsOf: sanitizedCChars)

        let newTokenStr: String
        if let validString = String(validating: temporaryInvalidCChars + [0], as: UTF8.self) {
            newTokenStr = validString
            temporaryInvalidCChars.removeAll()
        } else if let suffixIndex = temporaryInvalidCChars.firstIndex(where: { $0 != 0 }),
                  let validSuffix = String(validating: Array(temporaryInvalidCChars.suffix(from: suffixIndex)) + [0],
                                           as: UTF8.self) {
            newTokenStr = validSuffix
            temporaryInvalidCChars.removeAll()
        } else {
            newTokenStr = ""
        }

        batch.clear()
        batch.add(token: newToken, position: generatedTokenAccount, seqIDs: [0], logit: true)
        generatedTokenAccount += 1

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        return newTokenStr
    }

    private func tokenToCChars(token: llama_token) -> [CChar] {
        var length: Int32 = 8
        var piece = Array<CChar>(repeating: 0, count: Int(length))

        let nTokens = llama_token_to_piece(llama_model_get_vocab(model), token, &piece, length, 0, false)
        if nTokens >= 0 {
            return Array(piece.prefix(Int(nTokens)))
        } else {
            length = -nTokens
            piece = Array<CChar>(repeating: 0, count: Int(length))
            let nNewTokens = llama_token_to_piece(llama_model_get_vocab(model), token, &piece, length, 0, false)
            return Array(piece.prefix(Int(nNewTokens)))
        }
    }

    private func tokenize(text: String, addBos: Bool) -> [Token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (addBos ? 1 : 0) + 1
        
        return Array(unsafeUninitializedCapacity: n_tokens) { buffer, initializedCount in
            initializedCount = Int(
                llama_tokenize(llama_model_get_vocab(model), text, Int32(utf8Count), buffer.baseAddress, Int32(n_tokens), addBos, false)
            )
        }
    }

    /// 任意テキストのトークン数を返します。
    /// - Parameters:
    ///   - text: 計測対象テキスト
    ///   - addBos: 先頭に BOS を付与するか（デフォルトは呼び出し側で決定）
    /// - Returns: トークン数
    func countTokens(text: String, addBos: Bool) -> Int {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (addBos ? 1 : 0) + 1
        var tokenCount = 0
        _ = Array<Token>(unsafeUninitializedCapacity: n_tokens) { buffer, initializedCount in
            initializedCount = Int(
                llama_tokenize(
                    llama_model_get_vocab(model),
                    text,
                    Int32(utf8Count),
                    buffer.baseAddress,
                    Int32(n_tokens),
                    addBos,
                    false
                )
            )
            tokenCount = initializedCount
        }
        return tokenCount
    }

    func clear() {
        tokens.removeAll()
        temporaryInvalidCChars.removeAll()
        let mem = llama_get_memory(context)
        llama_memory_clear(mem, true)
    }

    /// 協調キャンセル用。次回ループ判定で停止させる。
    func stop() {
        ended = true
    }

    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
    }
}
