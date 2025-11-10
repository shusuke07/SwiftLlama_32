import Foundation

public enum SwiftLlamaError: Error {
    case decodeError
    case cancelled
    case others(String)
}
