import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Splash

/// The window's toolbar and the Chat's sidebar.
@MainActor
@Suite(.serialized)
struct MainViewTests {
    /// The sidebar shows or hides at launch as it did when the app last quit.
    @Test func theSidebarIsKeptFromOneLaunchToTheNext() {
        let kept = UserDefaults.standard.object(forKey: Preference.sidebar)
        defer { UserDefaults.standard.set(kept, forKey: Preference.sidebar) }
        let (state, folder) = Self.state()
        defer { try? FileManager.default.removeItem(at: folder) }
        state.showsSidebar = false
        #expect(!Self.state().0.showsSidebar)
        state.showsSidebar = true
        #expect(Self.state().0.showsSidebar)
    }

    /// The sidebar hidden takes the keyboard from its search field, where what is typed would
    /// filter a list no one sees.
    @Test func hidingTheSidebarTakesTheKeyboardFromIt() async throws {
        let kept = UserDefaults.standard.object(forKey: Preference.sidebar)
        defer { UserDefaults.standard.set(kept, forKey: Preference.sidebar) }
        let (state, folder) = Self.state()
        defer { try? FileManager.default.removeItem(at: folder) }
        state.showsSidebar = true
        _ = NSApplication.shared
        let window = NSWindow(contentViewController: NSHostingController(rootView: WithSidebar {
            TextField("Search", text: .constant(""))
        } content: {
            Color.clear
        }.environment(state)))
        window.setContentSize(CGSize(width: 800, height: 600))
        window.setFrameOrigin(NSPoint(x: -30000, y: -30000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        func field(in view: NSView) -> NSTextField? {
            (view as? NSTextField).flatMap { $0.isEditable ? $0 : nil } ?? view.subviews.lazy.compactMap(field).first
        }
        try await Self.waitFor(window.contentView.flatMap(field) != nil)
        window.makeFirstResponder(try #require(window.contentView.flatMap(field)))
        try await Self.waitFor((window.firstResponder as? NSTextView)?.isFieldEditor == true)
        // SwiftUI sees where the keyboard went, as it does before the user's next key.
        window.contentView?.layoutSubtreeIfNeeded()
        window.displayIfNeeded()
        state.toggleSidebar()
        try await Self.waitFor(window.firstResponder === window)
    }

    /// The Server button names the model, with the step of its start, then its speed while it
    /// answers, and says Server while none runs.
    @Test func theServerButtonSaysWhatTheServerDoes() async throws {
        let server = try FakeServer()
        defer { server.remove() }
        let state = server.state
        let model = "example/Model"
        let name = state.displayName(of: model)
        #expect(state.serverTitle == "Server")
        state.server.start(model, settings: ServeSettings(values: ["--port": String(server.port)]))
        #expect(state.serverTitle == "\(name) · Starting")
        try await Self.waitFor(state.serverTitle == "\(name) · Loading weights")
        server.signal("loaded")
        try await Self.waitFor(state.serverTitle == "\(name) · Warming up")
        server.signal("warm")
        try await Self.waitFor(state.server.phase == .ready)
        #expect(state.serverTitle == name)
        server.signal("answering")
        try await Self.waitFor(state.serverTitle == "\(name) · 42 tok/s")
        state.server.stop()
        #expect(state.serverTitle == name)
        try await Self.waitFor(state.server.phase == .stopped)
        #expect(state.serverTitle == "Server")
    }

    /// An app's state with its own chats, apart from the user's.
    private static func state() -> (AppState, URL) {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-main-\(UUID().uuidString)")
        return (AppState(runtime: Showcase.runtime, chats: ChatStore(folder: folder), keychain: Keychain(service: "Splash Tests")), folder)
    }

    private static func waitFor(_ condition: @autoclosure () -> Bool, seconds: Double = 15,
                                sourceLocation: SourceLocation = #_sourceLocation) async throws {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while !condition() {
            try #require(ContinuousClock.now < deadline, "timed out", sourceLocation: sourceLocation)
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}

/// A runtime whose launcher starts as `splash serve` does, one step at a time as the test signals
/// it, then answers as a ready server; its status reports a request running once signalled so.
@MainActor
private final class FakeServer {
    let root: URL
    let state: AppState
    let port: Int

    init() throws {
        let runtime = try FakeRuntime("splash-fake-server", launcher: Self.script)
        root = runtime.root
        port = runtime.port
        state = runtime.state(keychain: Keychain(service: "Splash Tests"))
    }

    /// Lets the launcher go on past the step that waits for `name`.
    func signal(_ name: String) {
        FileManager.default.createFile(atPath: root.appending(path: name).path, contents: nil)
    }

    /// Stops what still runs and removes the runtime.
    func remove() {
        state.server.stop()
        try? FileManager.default.removeItem(at: root)
    }

    private static let script = #"""
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    port = int(sys.argv[sys.argv.index('--port') + 1])
    def after(name):
        while not (root / name).exists(): time.sleep(0.02)
    print('Loading · weights', flush=True)
    after('loaded')
    print('Weights loaded', flush=True)
    after('warm')
    lock = root / 'build/runtime' / f'serve-{port}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({'pid': os.getpid()}))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            body = {}
            if self.path == '/v1/models': body = {'data': [{}]}
            if self.path == '/status':
                running = [{}] if (root / 'answering').exists() else []
                body = {'active_requests': running, 'metrics': {'decode_tokens_per_second': 41.6}}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
    """#
}
