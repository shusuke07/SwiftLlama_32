import Foundation
import llama
import os

class LlamaModel {
    private let model: Model
    private let configuration: Configuration
    private let context: OpaquePointer
    private let sampler: UnsafeMutablePointer<llama_sampler>
    private var batch: Batch
    private var batchCapacity: Int32
    private var tokens: [Token]
    private var temporaryInvalidCChars: [CChar] = []
    private var generatedTokenAccount: Int32 = 0
    private var ended = false
    private let logger = Logger(subsystem: "SwiftLlama", category: "LlamaModel")
    private var debugLog: ((String) -> Void)?

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
        guard let loadedModel = llama_model_load_from_file(path, model_params) else {
            // 初期化失敗時でもバックエンドを解放
            llama_backend_free()
            throw SwiftLlamaError.others("Cannot load model at path \(path)")
        }
        // デコーダ有無チェック（デコード不可モデルを早期に排除）
        guard llama_model_has_decoder(loadedModel) else {
            llama_model_free(loadedModel)
            llama_backend_free()
            throw SwiftLlamaError.others("Model does not have a decoder")
        }
        // コンテキスト作成
        guard let createdContext = llama_init_from_model(loadedModel, configuration.contextParameters) else {
            llama_model_free(loadedModel)
            llama_backend_free()
            throw SwiftLlamaError.others("Cannot load model context")
        }
        // 事前チェック（self 代入前に失敗時の解放を徹底）
        let n_ctx = llama_n_ctx(createdContext)
        let n_ctx_train = llama_model_n_ctx_train(loadedModel)
        logger.info("[SwiftLlama][CTX_INFO] requested=\(configuration.nCTX, privacy: .public) effective=\(n_ctx, privacy: .public) train_limit=\(n_ctx_train, privacy: .public)")
        if n_ctx > n_ctx_train {
            llama_free(createdContext)
            llama_model_free(loadedModel)
            llama_backend_free()
            throw SwiftLlamaError.others("Model was trained on \(n_ctx_train) context but tokens \(n_ctx) specified")
        }
        // ここから self に反映
        self.model = loadedModel
        self.context = createdContext
        self.tokens = []
        let initialCapacity = Int32(configuration.batchSize * max(1, configuration.historyLimit) * 2)
        self.batch = llama_batch_init(initialCapacity, 0, 1)
        self.batchCapacity = initialCapacity
        self.sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
        // 推奨構成: top-k -> top-p -> temp -> dist（最後に選択器）
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(Int32(configuration.topK)))
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(configuration.topP, 1))
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(configuration.temperature))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32(configuration.seed)))
    }

    private func checkContextLength(context: Context, model: Model) throws {
        let n_ctx = llama_n_ctx(context)
        let n_ctx_train = llama_n_ctx_train(model)
        if n_ctx > n_ctx_train {
            throw SwiftLlamaError.others("Model was trained on \(n_ctx_train) context but tokens \(n_ctx) specified")
        }
    }

    func setDebugLogger(_ f: ((String) -> Void)?) {
        self.debugLog = f
    }

    func start(for prompt: Prompt) throws {
        ended = false
        tokens = tokenize(text: prompt.prompt, addBos: true)
        temporaryInvalidCChars = []
        // 初回プロンプトに応じてバッチ容量を安全に確保し直す
        if Int32(tokens.count) > batchCapacity {
            llama_batch_free(batch)
            batch = llama_batch_init(Int32(tokens.count), 0, 1)
            batchCapacity = Int32(tokens.count)
        } else {
            batch.clear()
        }
        if configuration.debugLogTokens {
            let tokenPieces: [String] = tokens.map { t in
                let bytes = tokenToCChars(token: t)
                return String(validating: bytes + [0], as: UTF8.self) ?? ""
            }
            debugLog?("LLM_TOKEN_INIT count=\(self.tokens.count)")
            for (i, t) in tokens.enumerated() {
                let piece = i < tokenPieces.count ? tokenPieces[i] : ""
                debugLog?("LLM_TOKEN_INIT_ITEM index=\(i) id=\(t) piece=\(piece)")
            }
        }

        // Use runtime-reported n_batch as the hard cap; fall back to config if needed
        let runtimeBatch = llama_n_batch(context)
        let maxBatchSize = runtimeBatch > 0 ? Int32(runtimeBatch) : Int32(configuration.batchSize)
        logger.info("[SwiftLlama][BATCH_INFO] requested=\(self.configuration.batchSize, privacy: .public) runtime=\(runtimeBatch, privacy: .public)")
        var processed = 0
        while processed < tokens.count {
            batch.clear()
            let remaining = tokens.count - processed
            let chunkSize = Int(min(Int32(remaining), maxBatchSize))
            for i in 0..<chunkSize {
                let globalIndex = processed + i
                let token = tokens[globalIndex]
                let isLastToken = (globalIndex == tokens.count - 1)
                batch.add(token: token, position: Int32(globalIndex), seqIDs: [0], logit: isLastToken)
            }
            if llama_decode(context, batch) != 0 {
                throw SwiftLlamaError.decodeError
            }
            processed += chunkSize
        }
        generatedTokenAccount = Int32(tokens.count)
    }

    func `continue`() throws -> String {
        let newToken =  llama_sampler_sample(sampler, context, batch.n_tokens - 1)
        if configuration.debugLogTokens {
            let bytes = tokenToCChars(token: newToken)
            let piece = String(validating: bytes + [0], as: UTF8.self) ?? ""
            debugLog?("LLM_TOKEN_GEN id=\(newToken) piece=\(piece)")
        }

        if llama_vocab_is_eog(llama_model_get_vocab(model), newToken) {
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
        let vocab = llama_model_get_vocab(model)
        var need = llama_tokenize(vocab, text, Int32(utf8Count), nil, 0, addBos, false)
        if need == 0 { return [] }
        var capacity = need < 0 ? Int(-need) : Int(need)
        if capacity <= 0 { capacity = max(utf8Count + 8, 8) }
        var result: [Token] = []
        var buf = UnsafeMutablePointer<llama_token>.allocate(capacity: capacity)
        defer { buf.deallocate() }
        var count = Int(llama_tokenize(vocab, text, Int32(utf8Count), buf, Int32(capacity), addBos, false))
        if count < 0 {
            let newCap = Int(-count)
            buf.deallocate()
            let newBuf = UnsafeMutablePointer<llama_token>.allocate(capacity: newCap)
            defer { newBuf.deallocate() }
            let ok = Int(llama_tokenize(vocab, text, Int32(utf8Count), newBuf, Int32(newCap), addBos, false))
            if ok > 0 {
                result = Array(UnsafeBufferPointer(start: newBuf, count: ok))
            } else {
                result = []
            }
            return result
        }
        result = Array(UnsafeBufferPointer(start: buf, count: count))
        return result
    }

    /// 任意テキストのトークン数を返します。
    /// - Parameters:
    ///   - text: 計測対象テキスト
    ///   - addBos: 先頭に BOS を付与するか（デフォルトは呼び出し側で決定）
    /// - Returns: トークン数
    func countTokens(text: String, addBos: Bool) -> Int {
        let utf8Count = text.utf8.count
        let vocab = llama_model_get_vocab(model)
        var need = llama_tokenize(vocab, text, Int32(utf8Count), nil, 0, addBos, false)
        if need < 0 { return Int(-need) }
        return Int(need)
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
        llama_model_free(model)
        llama_backend_free()
    }
}
