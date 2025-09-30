import Foundation
import llama
import Combine

public class SwiftLlama {
    private let model: LlamaModel
    private let configuration: Configuration
    private var contentStarted = false
    private var sessionSupport = false {
        didSet {
            if !sessionSupport {
                session = nil
            }
        }
    }

    private var session: Session?
    private lazy var resultSubject: CurrentValueSubject<String, Error> = {
        .init("")
    }()
    private var generatedTokenCache = ""

    var maxLengthOfStopToken: Int {
        configuration.stopTokens.map { $0.count }.max() ?? 0
    }

    public init(modelPath: String,
                 modelConfiguration: Configuration = .init()) throws {
        self.model = try LlamaModel(path: modelPath, configuration: modelConfiguration)
        self.configuration = modelConfiguration
    }

    /// 任意テキストのトークン数を返します。
    /// - Parameters:
    ///   - text: 計測対象の文字列
    ///   - addBos: 先頭に BOS を付与するか（デフォルト: true）
    /// - Returns: トークン数
    public func countTokens(of text: String, addBos: Bool = true) -> Int {
        model.countTokens(text: text, addBos: addBos)
    }

    private func prepare(sessionSupport: Bool, for prompt: Prompt) -> Prompt {
        contentStarted = false
        generatedTokenCache = ""
        self.sessionSupport = sessionSupport
        if sessionSupport {
            if session == nil {
                session = Session(lastPrompt: prompt)
            } else {
                session?.lastPrompt = prompt
            }
            // trim history to configuration.historyLimit
            if var sessionPrompt = session?.sessionPrompt {
                if configuration.historyLimit >= 0 {
                    let trimmed = Array(sessionPrompt.history.suffix(configuration.historyLimit))
                    sessionPrompt = Prompt(type: sessionPrompt.type,
                                           systemPrompt: sessionPrompt.systemPrompt,
                                           userMessage: sessionPrompt.userMessage,
                                           history: trimmed)
                }
                return sessionPrompt
            }
            return prompt
        } else {
            // trim provided prompt history when session is disabled as well
            if configuration.historyLimit >= 0 {
                let trimmed = Array(prompt.history.suffix(configuration.historyLimit))
                return Prompt(type: prompt.type,
                              systemPrompt: prompt.systemPrompt,
                              userMessage: prompt.userMessage,
                              history: trimmed)
            }
            return prompt
        }
    }

    private func isStopToken() -> Bool {
        configuration.stopTokens.reduce(false) { partialResult, stopToken in
            generatedTokenCache.hasSuffix(stopToken)
        }
    }

    private func response(for prompt: Prompt, output: (String) -> Void, finish: () -> Void) {
        func finaliseOutput() {
            configuration.stopTokens.forEach {
                generatedTokenCache = generatedTokenCache.replacingOccurrences(of: $0, with: "")
            }
            // NUL を最終出力前に除去
            if !generatedTokenCache.isEmpty {
                generatedTokenCache = generatedTokenCache.replacingOccurrences(of: "\u{0000}", with: "")
            }
            output(generatedTokenCache)
            finish()
            generatedTokenCache = ""
        }
        defer { model.clear() }
        var finishedEarly = false
        do {
            try model.start(for: prompt)
            while model.shouldContinue {
                if Task.isCancelled {
                    finishedEarly = true
                    finish()
                    break
                }
                var delta = try model.continue()
                // 受信直後に NUL を除去
                if !delta.isEmpty {
                    delta = delta.replacingOccurrences(of: "\u{0000}", with: "")
                }
                if contentStarted { // remove the prefix empty spaces
                    if needToStop(after: delta, output: output) {
                        finishedEarly = true
                        finish()
                        break
                    }
                } else {
                    delta = delta.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !delta.isEmpty {
                        contentStarted = true
                        if needToStop(after: delta, output: output) {
                            finishedEarly = true
                            finish()
                            break
                        }
                    }
                }
            }
        } catch {
            // no-op; finalization happens after do-catch
        }
        if !finishedEarly {
            finaliseOutput()
        } else {
            generatedTokenCache = ""
        }
    }

    /// Handling logic of StopToken
    private func needToStop(after delta: String, output: (String) -> Void) -> Bool {
        guard !configuration.stopTokens.isEmpty else {
            output(delta)
            return false
        }

        let maxLen = maxLengthOfStopToken
        guard maxLen > 0 else {
            output(delta)
            return false
        }

        // 逐次（文字単位）処理で固定長ウィンドウを維持
        for ch in delta {
            // ウィンドウが満杯なら先頭を確定出力して追い出す
            if generatedTokenCache.count == maxLen {
                if let first = generatedTokenCache.first {
                    output(String(first))
                    generatedTokenCache.removeFirst()
                }
            }

            // 新文字を追加
            generatedTokenCache.append(ch)

            // ウィンドウが満杯になってから検査開始
            if generatedTokenCache.count == maxLen {
                if let matchedLen = configuration.stopTokens
                    .compactMap({ token in generatedTokenCache.hasSuffix(token) ? token.count : nil })
                    .max() {
                    // バッファ先頭〜トークン直前を出力し停止
                    let keepCount = max(0, maxLen - matchedLen)
                    if keepCount > 0 {
                        let prefix = String(generatedTokenCache.prefix(keepCount))
                        if !prefix.isEmpty { output(prefix) }
                    }
                    generatedTokenCache = ""
                    return true
                }
            }
        }
        return false
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) -> AsyncThrowingStream<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        return .init { continuation in
            let task = Task {
                response(for: sessionPrompt) { [weak self] delta in
                    continuation.yield(delta)
                    self?.session?.response(delta: delta)
                } finish: { [weak self] in
                    continuation.finish()
                    self?.session?.endResponse()
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) -> AnyPublisher<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        Task {
            response(for: sessionPrompt) { delta in
                resultSubject.send(delta)
                session?.response(delta: delta)
            } finish: {
                resultSubject.send(completion: .finished)
                session?.endResponse()
            }
        }
        return resultSubject.eraseToAnyPublisher()
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) async throws -> String {
        var result = ""
        for try await value in start(for: prompt) {
            result += value
        }
        return result
    }
}
