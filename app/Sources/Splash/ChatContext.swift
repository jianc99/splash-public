import Foundation
import Observation

/// How much of the running model's context the next prompt takes. Thinking and the reply share
/// what it leaves: a prompt that leaves less than the reply allowance still goes, as a reply may be
/// short, but one the context cannot hold does not.
struct ContextUsage: Equatable, Sendable {
    /// How full the context is: room for a reply, less than the allowance, or no room at all.
    enum Level { case normal, tight, full }

    let input: Int
    let limit: Int

    /// The room a reply is meant to have: 4,096 tokens, or an eighth of a smaller context.
    var allowance: Int { min(4096, limit / 8) }
    var fits: Bool { input < limit }
    var level: Level { !fits ? .full : input > limit - allowance ? .tight : .normal }
    /// The share of the context the prompt takes, at most all of it.
    var fraction: Double { min(1, Double(input) / Double(max(1, limit))) }

    /// The exact figures: 12,345 of 262,144 tokens · 5%.
    var summary: String {
        let share = (Double(input) / Double(max(1, limit))).formatted(.percent.precision(.fractionLength(0)))
        return "\(input.formatted()) of \(limit.formatted()) tokens · \(share)"
    }

    /// What a prompt short of room means for the reply, nil while there is room.
    var note: String? {
        switch level {
        case .normal: nil
        case .tight: "\((limit - input).formatted()) tokens are left for thinking and the reply."
        case .full: "Too long for the model’s context. Start a new chat, or shorten the message."
        }
    }

    /// The figures and the note, as the composer's ring says them.
    var explanation: String {
        [summary, note].compactMap(\.self).joined(separator: "\n")
    }
}

/// Counts the next prompt, the open conversation and the draft as the model reads them, after a
/// pause in typing. A count belongs to its whole prompt: history, draft, images, model, reasoning
/// effort and context; one finishing after the prompt changed is dropped.
@MainActor @Observable
final class ChatContext {
    struct Request: Equatable, Sendable {
        let history: [ChatMessage]
        let text: String
        let attachments: [Attachment]
        let target: ChatTarget
        /// A restarted model may use another tokenizer even under the same name.
        var started: Date? = nil
        /// Where the attachments are kept (`ChatStore.attachments`).
        let files: URL
        /// The conversation whose prompt it is, nil for a new chat's.
        var conversation: UUID? = nil

        var messages: [ChatMessage] {
            history + (text.isEmpty && attachments.isEmpty ? []
                : [ChatMessage(role: .user, text: text, attachments: attachments.isEmpty ? nil : attachments)])
        }
    }

    typealias Counter = @Sendable (Request) async throws -> Int

    /// The last prompt counted in each conversation, by its id, nil for a new chat, and its count:
    /// a chat opened again shows its last count at once, while its next prompt is counted. With
    /// the last prompt of all alone, the ring went as each chat opened and came back once counted.
    private(set) var counted: [UUID?: (request: Request, usage: ContextUsage)] = [:]
    @ObservationIgnored private let counter: Counter
    @ObservationIgnored private let delay: Duration
    /// The prompt being counted, or waiting for typing to pause.
    @ObservationIgnored private var pending: Request?
    @ObservationIgnored private var task: Task<Void, Never>?

    init(delay: Duration = .milliseconds(300),
         counter: @escaping Counter = { try await ChatClient.countTokens($0.messages, with: $0.target, files: $0.files) }) {
        self.delay = delay
        self.counter = counter
    }

    /// The count of exactly `request`, nil until it is counted.
    func usage(for request: Request) -> ContextUsage? {
        counted[request.conversation].flatMap { $0.request == request ? $0.usage : nil }
    }

    /// The count to show for `conversation`, of `history`, with `target`: its last one, while the
    /// next is counted, a reply streams or another reasoning effort is chosen, as long as the
    /// conversation has only grown since by the draft counted and its reply, or had its last answer
    /// asked again; for a conversation a new chat's message started, the new chat's.
    func shown(in conversation: UUID?, history: [ChatMessage], target: ChatTarget, started: Date?) -> ContextUsage? {
        for key in [conversation, nil] {
            guard let counted = counted[key], counted.request.target.sameModel(as: target), counted.request.started == started
            else { continue }
            let base = counted.request.history
            let grown = history.dropFirst(base.count)
            // A new chat's count shows only on the chat its message started: other chats share its
            // empty history. Message ids tell any other conversation apart.
            let sent = history.starts(with: base) && grown.count <= 2 && (!base.isEmpty || grown.first.map {
                $0.role == .user && $0.text == counted.request.text && ($0.attachments ?? []) == counted.request.attachments
            } ?? true)
            let askedAgain = base.last?.role == .assistant && history.count == base.count
                && history.dropLast().elementsEqual(base.dropLast())
            if sent || askedAgain { return counted.usage }
        }
        return nil
    }

    /// Counts `request` once typing pauses, in place of the prompt counted before in its
    /// conversation; nil counts nothing. A count that fails leaves none there: the server checks
    /// every prompt it is sent.
    func refresh(_ request: Request?) {
        guard request != pending else { return }
        task?.cancel()
        task = nil
        pending = request
        guard let request, usage(for: request) == nil, let limit = request.target.context, limit > 0 else { return }
        let counter = counter, delay = delay
        task = Task { [weak self] in
            guard (try? await Task.sleep(for: delay)) != nil else { return }
            let count = try? await counter(request)
            guard let self, !Task.isCancelled, pending == request else { return }
            counted[request.conversation] = count.map { (request, ContextUsage(input: $0, limit: limit)) }
        }
    }
}

extension ChatTarget {
    /// Whether `other` is the same model, served the same way, whatever reasoning effort it asks for.
    func sameModel(as other: ChatTarget) -> Bool {
        url == other.url && model == other.model && acceptsImages == other.acceptsImages && context == other.context
            && apiKey == other.apiKey
    }
}

/// Reads the next prompt into the model's cache while it is written, so that the reply to a long
/// message starts as soon as it is sent: the prompt so far, once typing pauses for 1 s, as a request
/// for one token whose reply is dropped. The request is the chat's own, at its priority: the reply
/// to the message sent meanwhile waits for what it has read and goes on from there, where one of a
/// lower priority would be preempted and read again (Qwen3.8-27B MLX on an M5 Max, a 5.6K-token
/// message: its first token 8.39 s cold, 0.17 s once read ahead, 7.62 s sent 0.5 s into the read;
/// 9.29 s with the read at background priority). A prompt never sent costs what reading it costs,
/// about 1.5 s of the GPU a thousand tokens on that Mac, the text added since the last read alone
/// at each later pause, and its KV and one state in the cache, which go first when memory is short.
@MainActor
final class ReadAhead {
    typealias Reader = @Sendable (ChatContext.Request) async -> Void

    /// What a read must add to the last one to be sent: a cache page, 32 tokens, at about four
    /// characters a token; a smaller step reuses nothing new. Text of Chinese, Japanese or Korean,
    /// a token a character or two, is read in fewer, larger steps.
    static let step = 128

    private let reader: Reader
    private let delay: Duration
    /// The prompt read last, or being read, and the read in flight or waiting for the pause.
    private var read: ChatContext.Request?
    private var reading = false
    private var task: Task<Void, Never>?

    init(delay: Duration = .seconds(1), reader: @escaping Reader = { await ChatClient.readAhead($0) }) {
        self.delay = delay
        self.reader = reader
    }

    /// Takes the next prompt as it changes, nil while there is none to read: no model answers, a
    /// reply streams or nothing is written. A read in flight goes on while the prompt only grows
    /// past it, or is sent, and stops when its text is no longer the start of the prompt's (an
    /// edit inside it), or the chat or the model changes.
    func refresh(_ request: ChatContext.Request?) {
        if reading, let read, let request, !request.continues(read) {
            task?.cancel()
            reading = false
            self.read = nil
        }
        guard !reading else { return }
        task?.cancel()
        guard let request else { return }
        let base = read.flatMap { request.continues($0) ? $0.text.count : nil } ?? 0
        guard request.text.count - base >= Self.step else { return }
        let reader = reader, delay = delay
        task = Task { [weak self] in
            guard (try? await Task.sleep(for: delay)) != nil, let self else { return }
            read = request
            reading = true
            await reader(request)
            if read == request { reading = false }
        }
    }
}

extension ChatContext.Request {
    /// Whether this prompt goes on from `other`: the same chat, model, history and attachments, and
    /// its text starting with `other`'s.
    func continues(_ other: Self) -> Bool {
        conversation == other.conversation && target == other.target && started == other.started
            && history == other.history && attachments == other.attachments && text.hasPrefix(other.text)
    }
}
