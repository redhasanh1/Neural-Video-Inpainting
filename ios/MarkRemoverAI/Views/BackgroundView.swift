import AVKit
import PhotosUI
import Combine
import SwiftUI

@MainActor
final class BackgroundModel: ObservableObject {
    enum Stage: Equatable {
        case picking, selecting, working(String), done(URL), failed(String)
    }

    @Published var stage: Stage = .picking
    @Published var frame: VideoFrame?
    @Published var selections: [Selection] = []
    @Published var isPreviewingMask = false
    /// The pipeline tracks each object_id separately, so a second subject needs
    /// its own id rather than more points on the first.
    @Published var currentObject = 0
    /// Hand-corrections to the SAM2 mask. The worker takes these in place of
    /// what it generated, so a bad edge can be fixed before tracking runs.
    let maskBuilder = StaticMaskBuilder()

    /// SwiftUI only observes the object a view declares. maskBuilder is nested
    /// inside this one, so its updates never reached the canvas — the drawing
    /// was correct but the screen stayed stale until something else forced a
    /// redraw. Forwarding its change events fixes that.
    private var maskBuilderChanges: AnyCancellable?

    init() {
        maskBuilderChanges = maskBuilder.objectWillChange.sink { [weak self] in
            self?.objectWillChange.send()
        }
    }
    @Published var settings = BackgroundSettings()
    @Published private(set) var sourceURL: URL?

    /// The background pipeline allows ten minutes and ten seconds, matching
    /// the website's own pre-upload check.
    static let maxDuration: Double = 610

    @Published private(set) var estimatedCredits: Double = 1
    @Published private(set) var duration: Double = 0
    private var pollTask: Task<Void, Never>?

    var points: [SelectionPoint] { selections.flatMap(\.points) }
    var canProcess: Bool { frame != nil && !points.isEmpty }
    var objectCount: Int { selections.count }
    private var previewTask: Task<Void, Never>?


    deinit { pollTask?.cancel() }

    func load(videoURL: URL) async {
        reset()
        sourceURL = videoURL
        duration = await VideoFrameExtractor.duration(of: videoURL)

        guard duration <= Self.maxDuration else {
            let seconds = Int(duration.rounded())
            stage = .failed("That clip is \(seconds / 60)m \(seconds % 60)s. The limit here is 10 minutes 10 seconds.")
            return
        }

        do {
            let first = try await VideoFrameExtractor.frame(from: videoURL)
            frame = first
            maskBuilder.begin(videoSize: first.pixelSize)
            estimatedCredits = CreditEstimate.credits(
                duration: duration, size: first.pixelSize, isBackground: true
            )
            stage = .selecting
        } catch {
            stage = .failed(error.localizedDescription)
        }
    }

    func reset() {
        pollTask?.cancel(); pollTask = nil
        stage = .picking
        frame = nil
        previewTask?.cancel(); previewTask = nil
        selections = []
        currentObject = 0
        maskBuilder.clear()
        sourceURL = nil
        duration = 0
    }

    func addPoint(normalized: CGPoint, label: Int = 1) {
        guard let frame else { return }
        let point = SelectionPoint(
            x: Int((normalized.x * frame.pixelSize.width).rounded()),
            y: Int((normalized.y * frame.pixelSize.height).rounded()),
            label: label,
            objectId: currentObject
        )

        if let index = selections.firstIndex(where: { $0.id == currentObject }) {
            selections[index].points.append(point)
        } else {
            var selection = Selection(id: currentObject)
            selection.points = [point]
            selections.append(selection)
        }
        refreshPreview()
    }

    func nextObject() {
        guard selections.contains(where: { $0.id == currentObject && !$0.points.isEmpty }) else { return }
        currentObject = (selections.map(\.id).max() ?? 0) + 1
    }

    func undo() {
        guard let index = selections.firstIndex(where: { $0.id == currentObject })
                ?? selections.indices.last else { return }
        selections[index].points.removeLast()
        if selections[index].points.isEmpty {
            let removed = selections.remove(at: index).id
            if currentObject == removed { currentObject = selections.last?.id ?? 0 }
        } else {
            refreshPreview()
        }
    }

    /// Mirrors the website: only the current object's points are previewed, so
    /// one object's mask never bleeds into another's.
    private func refreshPreview() {
        guard let frame,
              let selection = selections.first(where: { $0.id == currentObject }),
              !selection.points.isEmpty else { return }

        previewTask?.cancel()
        previewTask = Task { [weak self] in
            await MainActor.run { self?.isPreviewingMask = true }
            defer { Task { @MainActor in self?.isPreviewingMask = false } }

            guard let payload = VideoFrameExtractor.previewPayload(
                frame: frame, points: selection.points
            ) else { return }

            // Swift flattens try? over an optional return, so this is already
            // a plain String by the time it binds.
            guard let mask = try? await APIClient.shared.previewMask(
                frameBase64PNG: payload.base64PNG,
                frameIndex: frame.frameIndex,
                points: payload.points,
                videoWidth: Int(payload.size.width),
                videoHeight: Int(payload.size.height)
            ) else { return }

            let tinted = MaskRenderer.tinted(
                base64: mask, color: SelectionPalette.uiColor(selection.colorIndex)
            )
            await MainActor.run {
                guard let self,
                      let index = self.selections.firstIndex(where: { $0.id == selection.id })
                else { return }
                self.selections[index].mask = tinted
            }
        }
    }

    func run(appState: AppState) async {
        guard let sourceURL, let frame else { return }
        guard appState.credits >= estimatedCredits else {
            stage = .failed(APIError.outOfCredits.localizedDescription)
            return
        }

        do {
            stage = .working("Uploading…")
            let jobId = try await APIClient.shared.backgroundUpload(
                fileURL: sourceURL,
                filename: sourceURL.lastPathComponent,
                contentType: "video/mp4",
                width: Int(frame.pixelSize.width),
                height: Int(frame.pixelSize.height),
                fps: frame.fps,
                duration: duration
            )

            stage = .working("Finding the subject…")
            try await APIClient.shared.backgroundSelect(
                jobId: jobId, points: points, frameIndex: frame.frameIndex
            )

            stage = .working("Following it through the clip…")
            var corrections: [(objectId: Int, base64: String)] = []
            if !maskBuilder.isEmpty, let drawn = maskBuilder.exportBase64PNG() {
                corrections.append((objectId: currentObject, base64: drawn))
                AppLog.info(.editor, "Sending a hand-corrected mask for object \(currentObject)")
            }
            try await APIClient.shared.backgroundTrack(jobId: jobId, modifiedMasks: corrections)

            // Tracking has to finish before export: the masks it produces are
            // exactly what export consumes. Firing export straight after track
            // asks the worker to composite masks that do not exist yet.
            try await waitFor(jobId: jobId, status: "completed", label: "Tracking")

            stage = .working("Applying the background…")
            AppLog.info(.editor, "Background job \(jobId): \(settings.operation.rawValue) / \(settings.fill.rawValue), dilation \(Int(settings.dilation))")
            try await APIClient.shared.backgroundExport(jobId: jobId, settings: settings)

            JobStore.shared.record(id: jobId, kind: .background)
            poll(jobId: jobId, appState: appState)
        } catch {
            AppLog.error(.editor, "Background failed: \(error.localizedDescription)")
            stage = .failed(error.localizedDescription)
        }
    }

    /// Polls until the job reports the status we're waiting on. Mirrors the
    /// website's two-phase wait: tracking, then export.
    private func waitFor(jobId: String, status target: String, label: String) async throws {
        let deadline = Date().addingTimeInterval(10 * 60)

        while Date() < deadline {
            try Task.checkCancellation()
            guard let status = try? await APIClient.shared.backgroundStatus(jobId: jobId) else {
                try await Task.sleep(for: .seconds(3))
                continue
            }

            if status.status == target { 
                AppLog.info(.editor, "\(label) complete")
                return
            }
            if status.status == "error" || status.status == "failed" {
                throw APIError.http(500, status.error ?? "\(label) failed on the worker.")
            }

            await MainActor.run {
                let percent = status.progress.map { " \($0)%" } ?? ""
                self.stage = .working("\(label)\(percent)…")
            }
            try await Task.sleep(for: .seconds(3))
        }
        throw APIError.http(504, "\(label) timed out.")
    }

    private func poll(jobId: String, appState: AppState) {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            let deadline = Date().addingTimeInterval(20 * 60)
            while !Task.isCancelled && Date() < deadline {
                try? await Task.sleep(for: .seconds(3))
                guard let status = try? await APIClient.shared.backgroundStatus(jobId: jobId) else { continue }

                switch status.status {
                case "export_complete":
                    let url = APIClient.shared.backgroundDownloadURL(jobId: jobId)
                    await MainActor.run { JobStore.shared.finish(id: jobId, resultURL: url.absoluteString) }
                    if let local = try? await APIClient.shared.download(url) {
                        await MainActor.run { self?.stage = .done(local) }
                    } else {
                        await MainActor.run { self?.stage = .failed("Finished, but the download failed.") }
                    }
                    await appState.refreshUser()
                    return
                case "failed", "error":
                    await MainActor.run {
                        JobStore.shared.fail(id: jobId, reason: status.error)
                        self?.stage = .failed(status.error ?? "The worker rejected it.")
                    }
                    return
                default:
                    await MainActor.run {
                        self?.stage = .working(status.message ?? "Working…")
                    }
                }
            }
            await MainActor.run { self?.stage = .failed("Timed out after 20 minutes.") }
        }
    }
}

struct BackgroundView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var model = BackgroundModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var markMode = 1
    @State private var tool: BackgroundTool = .click
    @State private var brushSize: Double = 0.04

    /// Click places SAM2 points; the rest paint a correction over the mask.
    enum BackgroundTool: String, CaseIterable, Identifiable {
        case click, rectangle, brush, eraser
        var id: String { rawValue }
        var title: String {
            switch self {
            case .click: return "Click"
            case .rectangle: return "Box"
            case .brush: return "Brush"
            case .eraser: return "Erase"
            }
        }
        var symbol: String {
            switch self {
            case .click: return "hand.tap"
            case .rectangle: return "rectangle"
            case .brush: return "paintbrush.pointed"
            case .eraser: return "eraser"
            }
        }
        var drawTool: StaticMaskBuilder.Tool? {
            switch self {
            case .click: return nil
            case .rectangle: return .rectangle
            case .brush: return .brush
            case .eraser: return .eraser
            }
        }
    }

    var body: some View {
        NavigationStack {
            Group {
                switch model.stage {
                case .picking: picker
                case .selecting: selector
                case .working(let message): busy(message)
                case .done(let url): result(url)
                case .failed(let message): failure(message)
                }
            }
            .navigationTitle("Background")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if model.frame != nil {
                    ToolbarItem(placement: .topBarTrailing) {
                        PhotosPicker(selection: $pickerItem, matching: .videos, photoLibrary: .shared()) {
                            Image(systemName: "arrow.triangle.2.circlepath")
                                .font(.subheadline)
                        }
                    }
                }
            }
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task {
                guard let movie = try? await item.loadTransferable(type: VideoFile.self) else {
                    model.stage = .failed("Couldn't read that video.")
                    return
                }
                await model.load(videoURL: movie.url)
            }
        }
    }

    private var picker: some View {
        ScrollView {
            VStack(spacing: 20) {
                ZStack {
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(Theme.orangeGradient)
                    VStack(spacing: 12) {
                        Image(systemName: "person.and.background.dotted")
                            .font(.system(size: 46))
                            .foregroundStyle(.white)
                        Text("Keep the subject,\nreplace everything else")
                            .font(.title3.bold())
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.vertical, 26)
                }
                .frame(height: 200)
                .padding(.horizontal, 16)

                PhotosPicker(selection: $pickerItem, matching: .videos, photoLibrary: .shared()) {
                    Label("Choose a video", systemImage: "photo.on.rectangle.angled")
                }
                .buttonStyle(PrimaryButtonStyle(gradient: Theme.orangeGradient))
                .padding(.horizontal, 16)

                Text("Up to 10 minutes")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 14)
        }
        .background(Color(.systemGroupedBackground))
    }

    private var selector: some View {
        ScrollView {
            VStack(spacing: 0) {
            if let frame = model.frame {
                EstimateBar(
                    duration: model.duration,
                    size: frame.pixelSize,
                    credits: model.estimatedCredits,
                    isBackground: true,
                    tint: Theme.orange
                )
                .padding(.horizontal, 16)
                .padding(.top, 10)

                VideoCanvas(
                    frame: frame,
                    selections: model.selections,
                    activeSelectionID: model.currentObject,
                    isBusy: model.isPreviewingMask,
                    drawnMask: model.maskBuilder.preview,
                    isDrawing: tool != .click,
                    onDraw: { from, to in
                        guard let drawTool = tool.drawTool else { return }
                        model.maskBuilder.stroke(
                            from: from, to: to, tool: drawTool, brushFraction: brushSize
                        )
                    },
                    onDrawEnded: { model.maskBuilder.endStroke() },
                    onTap: { location in
                        guard tool == .click else { return }
                        model.addPoint(normalized: location, label: markMode)
                    }
                )
                .frame(maxHeight: 300)
                .padding(.horizontal, 16)
                .padding(.top, 8)
            }

            stepHeader(1, "Mark the subject", "Point at what matters. This only builds a selection.")
                .padding(.top, 16)

            VStack(spacing: 12) {
            HStack(spacing: 8) {
                ForEach(BackgroundTool.allCases) { option in
                    Button {
                        tool = option
                        Haptics.tick()
                    } label: {
                        VStack(spacing: 3) {
                            Image(systemName: option.symbol).font(.subheadline)
                            Text(option.title).font(.caption2)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .fill(tool == option ? Theme.orangeSoft : Color(.tertiarySystemFill))
                        )
                        .foregroundStyle(tool == option ? Theme.orange : .primary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)

            if tool == .click {
                VStack(spacing: 5) {
                    // Plus and minus, not words: this is the same control
                    // every photo editor uses and it needs no explaining.
                    HStack(spacing: 10) {
                        ForEach([1, 0], id: \.self) { mode in
                            Button {
                                markMode = mode
                                Haptics.tick()
                            } label: {
                                HStack(spacing: 7) {
                                    Image(systemName: mode == 1 ? "plus.circle.fill" : "minus.circle.fill")
                                        .font(.title3)
                                    Text(mode == 1 ? "this" : "not this")
                                        .font(.subheadline.weight(.medium))
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 11)
                                .background(
                                    RoundedRectangle(cornerRadius: 11, style: .continuous)
                                        .fill(markMode == mode
                                              ? (mode == 1 ? Theme.positive.opacity(0.18) : Color.red.opacity(0.15))
                                              : Color(.tertiarySystemFill))
                                )
                                .foregroundStyle(markMode == mode
                                                 ? (mode == 1 ? Theme.positive : .red)
                                                 : .primary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 10)
            } else {
                HStack(spacing: 10) {
                    Text(tool == .eraser ? "Eraser size" : "Brush size")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 74, alignment: .leading)
                    Slider(value: $brushSize, in: 0.01...0.25).tint(Theme.orange)
                    Button("Clear") { model.maskBuilder.clear() }
                        .font(.caption)
                        .buttonStyle(.bordered)
                        .disabled(model.maskBuilder.isEmpty)
                }
                .padding(.horizontal, 16)
                .padding(.top, 10)
            }

            }
            .padding(13)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .padding(.horizontal, 16)
            .padding(.top, 10)

            Text(model.points.isEmpty
                 ? (model.settings.operation == .keepObject
                    ? "Tap what you want to keep."
                    : "Tap what you want gone.")
                 : "\(model.points.count) tap\(model.points.count == 1 ? "" : "s") across \(model.objectCount) subject\(model.objectCount == 1 ? "" : "s").")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
                .padding(.top, 12)

            options
                .padding(.top, 12)

            Spacer(minLength: 12)

            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    Button("Undo") { model.undo() }.disabled(model.points.isEmpty)
                    Button {
                        model.nextObject()
                        Haptics.tick()
                    } label: {
                        Label("Next subject", systemImage: "plus.circle")
                    }
                    .disabled(model.points.isEmpty)
                    Button("New video") { model.reset(); pickerItem = nil }
                }
                .font(.subheadline)
                .buttonStyle(.bordered)

            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
            }
        }
        .background(Color(.systemGroupedBackground))
        .safeAreaInset(edge: .bottom) {
            // Pinned: the thing you came here to press should never require
            // scrolling to find.
            Button((model.settings.operation == .keepObject ? "Replace background" : "Remove selection")
                   + " · \(CreditEstimate.label(model.estimatedCredits)) credit\(model.estimatedCredits == 1 ? "" : "s")") {
                Task { await model.run(appState: appState) }
            }
            .buttonStyle(PrimaryButtonStyle(enabled: model.canProcess, gradient: Theme.orangeGradient))
            .disabled(!model.canProcess)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.bar)
        }
    }

    /// Everything the website exposes: what survives, what fills the rest, and
    /// the two sliders that decide how clean the cut-out looks.
    private var options: some View {
        VStack(spacing: 14) {
            stepHeader(2, "Choose the result", "What the finished video looks like.")

            VStack(spacing: 14) {
            VStack(spacing: 6) {
                Text("The thing you marked")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Picker("Operation", selection: $model.settings.operation) {
                    ForEach(BackgroundOperation.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                Text(model.settings.operation.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                ForEach(BackgroundFill.allCases) { option in
                    Button {
                        model.settings.fill = option
                        Haptics.tick()
                    } label: {
                        VStack(spacing: 5) {
                            Image(systemName: option.symbol).font(.title3)
                            Text(option.title).font(.caption2.weight(.medium))
                        }
                        .frame(maxWidth: .infinity, minHeight: 62)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(model.settings.fill == option ? Theme.orangeSoft : Color(.tertiarySystemFill))
                        )
                        .foregroundStyle(model.settings.fill == option ? Theme.orange : .primary)
                    }
                    .buttonStyle(.plain)
                }
            }

            if model.settings.fill == .color {
                ColorPicker("Background colour", selection: $model.settings.color, supportsOpacity: false)
                    .font(.subheadline)
            }

            if model.settings.fill == .blur {
                slider("Blur", value: $model.settings.blurAmount, range: 5...50)
            }

            // Dilation grows the mask outward. A pixel or two removes the halo
            // of old background that otherwise clings to the edges.
            slider("Edge grow", value: $model.settings.dilation, range: 0...20)
            }
            .padding(13)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
        .padding(.horizontal, 16)
    }

    /// Numbered so the two groups read as sequential steps rather than two
    /// interchangeable settings sitting on top of each other.
    private func stepHeader(_ number: Int, _ title: String, _ subtitle: String) -> some View {
        HStack(alignment: .top, spacing: 9) {
            Text("\(number)")
                .font(.caption.bold())
                .foregroundStyle(.white)
                .frame(width: 20, height: 20)
                .background(Circle().fill(Theme.orange))
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
    }

    private func slider(_ label: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 68, alignment: .leading)
            Slider(value: value, in: range, step: 1).tint(Theme.orange)
            Text("\(Int(value.wrappedValue))")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 24, alignment: .trailing)
        }
    }

    private func busy(_ message: String) -> some View {
        VStack(spacing: 16) {
            Spacer()
            ProgressView().scaleEffect(1.5).tint(Theme.orange)
            Text(message).font(.subheadline).foregroundStyle(.secondary)
            Spacer()
        }
    }

    private func result(_ url: URL) -> some View {
        VStack(spacing: 14) {
            VideoPlayer(player: AVPlayer(url: url))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .padding(.horizontal, 16)
                .padding(.top, 8)

            if model.settings.fill == .transparent {
                Label("WebM with alpha, because MP4 cannot carry transparency.", systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 8)

            VStack(spacing: 10) {
                ShareLink(item: url) { Text("Share or save").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent)
                Button("Do another") { model.reset(); pickerItem = nil }
                    .font(.subheadline)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
    }

    private func failure(_ message: String) -> some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 40)).foregroundStyle(Theme.warning)
            Text(message)
                .font(.subheadline).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).padding(.horizontal, 30)
            Button("Start over") { model.reset(); pickerItem = nil }
                .buttonStyle(PrimaryButtonStyle(gradient: Theme.orangeGradient))
                .padding(.horizontal, 30)
            Spacer()
        }
    }
}
