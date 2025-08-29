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
            output(generatedTokenCache)
            finish()
            generatedTokenCache = ""
        }
        defer { model.clear() }
        var finishedEarly = false
        do {
            try model.start(for: prompt)
            while model.shouldContinue {
                var delta = try model.continue()
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
        generatedTokenCache += delta

        // 1) 完全一致のストップトークンを探索（最も早い出現位置を優先）
        if let (range, _) = configuration.stopTokens
            .compactMap({ token -> (Range<String.Index>, String)? in
                if let r = generatedTokenCache.range(of: token) { return (r, token) }
                return nil
            })
            .min(by: { $0.0.lowerBound < $1.0.lowerBound }) {
            let outputCandidate = String(generatedTokenCache[..<range.lowerBound])
            if !outputCandidate.isEmpty { output(outputCandidate) }
            generatedTokenCache = ""
            return true
        }

        // 2) 未検出の場合、末尾に残すべき最長のプレフィックス一致長を計算
        let maxLen = maxLengthOfStopToken
        let text = generatedTokenCache
        let textCount = text.count
        let maxCheck = min(textCount, maxLen)
        var keepLength = 0
        if maxCheck > 0 {
            for l in 1...maxCheck {
                let start = text.index(text.endIndex, offsetBy: -l)
                let suffix = text[start...]
                if configuration.stopTokens.contains(where: { $0.hasPrefix(suffix) }) {
                    keepLength = l
                }
            }
        }

        // 先頭側（確定出力分）を吐き出し、末尾の未確定部分だけ保持
        let emitCount = max(0, generatedTokenCache.count - keepLength)
        if emitCount > 0 {
            let emit = String(generatedTokenCache.prefix(emitCount))
            output(emit)
            generatedTokenCache.removeFirst(emit.count)
        }
        return false
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) -> AsyncThrowingStream<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        return .init { continuation in
            Task {
                response(for: sessionPrompt) { [weak self] delta in
                    continuation.yield(delta)
                    self?.session?.response(delta: delta)
                } finish: { [weak self] in
                    continuation.finish()
                    self?.session?.endResponse()
                }
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
