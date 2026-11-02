import Darwin
import Foundation
import Testing

@testable import Splash

/// A runtime of a test's own, in a temporary folder, whose launcher is a Python script that the
/// system's Python runs, and a loopback port nothing serves, for the script to serve on.
struct FakeRuntime {
    let root: URL
    let port: Int

    init(_ name: String, launcher script: String) throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "\(name)-\(UUID().uuidString)")
        let manager = FileManager.default
        for folder in ["install", ".venv/bin", "chats"] {
            try manager.createDirectory(at: root.appending(path: folder), withIntermediateDirectories: true)
        }
        // /usr/bin/python3 can be Apple's launcher; resolve the real interpreter before linking
        // it under the name `python`, which the launcher otherwise tries to find on PATH.
        let resolver = Process()
        resolver.executableURL = URL(filePath: "/usr/bin/python3")
        resolver.arguments = ["-c", "import sys; print(sys.executable)"]
        let output = Pipe()
        resolver.standardOutput = output
        try resolver.run()
        let executable = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        resolver.waitUntilExit()
        try #require(resolver.terminationStatus == 0 && !executable.isEmpty)
        try manager.createSymbolicLink(at: root.appending(path: ".venv/bin/python"), withDestinationURL: URL(filePath: executable))
        try script.write(to: root.appending(path: "install/launcher.py"), atomically: true, encoding: .utf8)
        let socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        defer { Darwin.close(socket) }
        var address = sockaddr_in(sin_len: UInt8(MemoryLayout<sockaddr_in>.size), sin_family: sa_family_t(AF_INET),
                                  sin_port: 0, sin_addr: in_addr(s_addr: inet_addr("127.0.0.1")), sin_zero: (0, 0, 0, 0, 0, 0, 0, 0))
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                _ = bind(socket, $0, length)
                _ = getsockname(socket, $0, &length)
            }
        }
        self.root = root
        port = Int(UInt16(bigEndian: address.sin_port))
    }

    /// The folder of the chats of `state`.
    var chats: URL { root.appending(path: "chats") }

    /// An app's state on this runtime, its server's log in the runtime's folder.
    @MainActor func state(keychain: Keychain) -> AppState {
        let runtime = Runtime(root: root)
        return AppState(runtime: runtime, chats: ChatStore(folder: chats), keychain: keychain,
                        server: Server(runtime: runtime, log: root.appending(path: "server.log")))
    }
}
