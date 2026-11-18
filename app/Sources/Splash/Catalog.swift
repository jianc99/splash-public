import Foundation

/// The models Splash suggests: `install/suggested-models.tsv`, which shell completion reads too.
/// Each line is `MODEL<TAB>NAME<TAB>MEMORY<TAB>SIZE<TAB>ABOUT`: the ID `splash serve --model`
/// takes, the model's name, the smallest Mac memory in GB the variant is meant for, its download
/// in GB, and a few words on the model. Best first.
struct Catalog {
    struct Variant: Hashable {
        let id: String
        let name: String
        let memory: Int
        /// The download, in GB.
        let size: Int
        let about: String
    }

    let variants: [Variant]

    init(contentsOf url: URL) {
        self.init(text: (try? String(contentsOf: url, encoding: .utf8)) ?? "")
    }

    init(text: String) {
        variants = text.split(separator: "\n").compactMap { line in
            let fields = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard !line.hasPrefix("#"), fields.count == 5, let memory = Int(fields[2]), let size = Int(fields[3])
            else { return nil }
            return Variant(id: fields[0], name: fields[1], memory: memory, size: size, about: fields[4])
        }
    }

    /// For each model, best first, the variant for a Mac with `memory` GB: the one meant for the
    /// most memory that Mac has, the first listed among equals. Models it has too little memory for
    /// are left out.
    func suggestions(memory: Int) -> [Variant] {
        var chosen: [String: Variant] = [:]
        for variant in variants where variant.memory <= memory {
            if variant.memory > chosen[variant.name]?.memory ?? -1 { chosen[variant.name] = variant }
        }
        var names: [String] = []
        for variant in variants where !names.contains(variant.name) { names.append(variant.name) }
        return names.compactMap { chosen[$0] }
    }

    /// The least memory any suggested model is meant for.
    var minimumMemory: Int? { variants.map(\.memory).min() }

    /// The name to show for model `id`: its suggested name, or else its repository's, without the
    /// variant and without the words after its name that say how its weights are stored, as
    /// "-128K-MLX-4bit-DWQ" does: "DeepSeek-R2-Distill-Qwen3.6-35B-A3B-Instruct", which names the
    /// model where its whole repository name, cut to fit, cut the model away.
    func name(of id: String) -> String {
        if let name = variants.first(where: { $0.id == id })?.name { return name }
        // Model files on this Mac: the file's or folder's name.
        if id.hasPrefix("/") { return URL(filePath: id).lastPathComponent.replacing(/(?i)\.gguf$/, with: "") }
        return Self.split(Self.repository(of: id)).name
    }

    /// What tells variants of a model apart: the folder of model files on this Mac, the GGUF variant
    /// `id` names, or the storage words its repository's name ends in ("MLX-4bit-DWQ"), or MLX for
    /// a model in MLX's format, or else its repository.
    static func variant(of id: String) -> String {
        if id.hasPrefix("/") { return URL(filePath: id).deletingLastPathComponent().lastPathComponent }
        if let variant = id.split(separator: ":", maxSplits: 1).dropFirst().first { return String(variant) }
        let storage = split(repository(of: id)).storage
        if !storage.isEmpty { return storage }
        return id.localizedCaseInsensitiveContains("mlx") ? "MLX" : String(id.split(separator: "/").last ?? Substring(id))
    }

    /// The repository's name of model `id`, without its owner and its variant.
    private static func repository(of id: String) -> String {
        id.split(separator: ":")[0].split(separator: "/").last.map(String.init) ?? id
    }

    /// `repository` split where the words that say how its weights are stored begin: its format,
    /// MLX or GGUF, which only such words follow, and those it ends in, a quantization ("4bit",
    /// "Q4_K_M", "DWQ", "bf16") or a context length ("128K"); the whole of it where nothing is left
    /// before them. Plain string checks: views ask for names on every render.
    static func split(_ repository: String) -> (name: String, storage: String) {
        var words = repository.split(separator: "-", omittingEmptySubsequences: false).map(String.init)
        var storage: [String] = []
        if let format = words.firstIndex(where: { ["mlx", "gguf"].contains($0.lowercased()) }), format > 0 {
            storage = Array(words[format...])
            words.removeSubrange(format...)
        }
        while words.count > 1, let last = words.last, isStorage(last) {
            storage.insert(words.removeLast(), at: 0)
        }
        return (words.joined(separator: "-"), storage.joined(separator: "-"))
    }

    /// Whether `word` of a repository's name says how its weights are stored: "4bit", "8bit",
    /// "bf16", "fp8", "nvfp4", "Q4_K_M", "IQ3_XS", "DWQ", "AWQ", "GPTQ", "QAT", "int4", or a context
    /// length, "128K".
    private static func isStorage(_ word: String) -> Bool {
        let lower = word.lowercased()
        if ["dwq", "awq", "gptq", "qat", "exl2", "mlx", "gguf"].contains(lower) { return true }
        if lower.hasSuffix("bit"), Int(lower.dropLast(3)) != nil { return true }
        if lower.hasSuffix("k"), Int(lower.dropLast()) != nil { return true }
        for prefix in ["bf", "fp", "nvfp", "mxfp", "int"] where lower.hasPrefix(prefix) {
            if Int(lower.dropFirst(prefix.count)) != nil { return true }
        }
        // A GGUF quantization: Q or IQ, a digit, then letters, digits and underscores.
        let quant = lower.hasPrefix("iq") ? lower.dropFirst(2) : lower.hasPrefix("q") ? lower.dropFirst() : nil
        if let quant, let first = quant.first, first.isNumber, quant.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" }) { return true }
        return false
    }

    /// The few words on model `id`, if it is a suggested one.
    func about(of id: String) -> String? {
        variants.first { $0.id == id }?.about
    }

    /// This Mac's unified memory in GB, as Apple sells it (24, 36, 48, ...).
    static var memory: Int {
        Int((Double(ProcessInfo.processInfo.physicalMemory) / Double(1 << 30)).rounded())
    }
}
