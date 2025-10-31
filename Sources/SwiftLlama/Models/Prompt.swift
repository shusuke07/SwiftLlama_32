import Foundation

public struct Prompt {
    public enum `Type` {
        case chatML
        case alpaca
        case llama
        case llama3
        case mistral
        case phi
        case gemma
        case qwen
        case baku
    }

    public let type: `Type`
    public let systemPrompt: String
    public let userMessage: String
    public let history: [Chat]

    public init(type: `Type`,
                systemPrompt: String = "",
                userMessage: String,
                history: [Chat] = []) {
        self.type = type
        self.systemPrompt = systemPrompt
        self.userMessage = userMessage
        self.history = history
    }

    public var prompt: String {
        switch type {
        case .llama: encodeLlamaPrompt()
        case .llama3: encodeLlama3Prompt()
        case .alpaca: encodeAlpacaPrompt()
        case .chatML: encodeChatMLPrompt()
        case .mistral: encodeMistralPrompt()
        case .phi: encodePhiPrompt()
        case .gemma: encodeGemmaPrompt()
        case .qwen: encodeQwenPrompt()
        case .baku: encodeBakuPrompt()
        }
    }

    private func encodeLlamaPrompt() -> String {
        """
        [INST]<<SYS>>
        \(systemPrompt)
        <</SYS>>
        \(history.map { $0.llamaPrompt }.joined())
        [/INST]
        [INST]
        \(userMessage)
        [/INST]
        """
    }

    private func encodeLlama3Prompt() -> String {
        let prompt = """
        <|start_header_id|>system<|end_header_id|>\(systemPrompt)<|eot_id|>
        
        \(history.map { $0.llama3Prompt }.joined())
        
        <|start_header_id|>user<|end_header_id|>\(userMessage)<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
      return prompt
    }

    private func encodeAlpacaPrompt() -> String {
        """
        Below is an instruction that describes a task.
        Write a response that appropriately completes the request.
        \(userMessage)
        """
    }

    private func encodeChatMLPrompt() -> String {
        """
        \(history.map { $0.chatMLPrompt }.joined())
        "<|im_start|>user"
        \(userMessage)<|im_end|>
        <|im_start|>assistant
        """
    }

    private func encodeMistralPrompt() -> String {
        """
        <s>
        \(history.map { $0.mistralPrompt }.joined())
        </s>
        [INST] \(userMessage) [/INST]
        """
    }

    private func encodePhiPrompt() -> String {
        """
        \(systemPrompt)
        \(history.map { $0.phiPrompt }.joined())
        <|user|>
        \(userMessage)
        <|end|>
        <|assistant|>
        """
    }

    private func encodeGemmaPrompt() -> String {
        """
        <start_of_turn>system
        \(systemPrompt)
        <end_of_turn>
        \(history.map { $0.gemmaPrompt }.joined())
        <start_of_turn>user
        \(userMessage)
        <end_of_turn>
        <start_of_turn>model
        """
    }

    private func encodeBakuPrompt() -> String {
        let historyPart = history.map { $0.bakuPrompt }.joined()
        let userCombined: String = {
            let sys = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !sys.isEmpty && history.isEmpty {
                return "\(sys)\n\n\(userMessage)"
            }
            return userMessage
        }()
        return """
        \(historyPart)
        <start_of_turn>user
        \(userCombined)
        <end_of_turn>
        <start_of_turn>model
        """
    }

    private func encodeQwenPrompt() -> String {
        """
        <|im_start|>system
        \(systemPrompt)<|im_end|>
        \(history.map { $0.qwenPrompt }.joined(separator: "\n"))
        <|im_start|>user
        \(userMessage)<|im_end|>
        <|im_start|>assistant
        """
    }
}
