import Combine
import SwiftUI

@MainActor
final class EditorModel: ObservableObject {
    enum Stage: Equatable {
        case picking
        case selecting
        case uploading
        case processing(progress: Int, message: String)
        case done(URL)
        case failed(String)
    }

    /// Two ways to say what should go, matching the website's split.
    enum Mode: String, CaseIterable, Identifiable {
        /// SAM2 tracks something that moves.
        case moving
        /// Something stationary: draw it once, repeated on every frame, no tracking.
        case fixed

        var id: String { rawValue }
        var title: String { self == .moving ? "Moves around" : "Stays in place" }
        var hint: String {
            self == .moving
                ? "Tap it once. It gets followed through the whole clip."
                : "Draw over it once. The same area is cleared on every frame."
        }
    }

    @Published var mode: Mode = .moving
    @Published var stage: Stage = .picking
    @Published var frame: VideoFrame?
    @Published var selections: [Selection] = []
    @Published var activeSelectionID: Int?
    @Published var isPreviewingMask = false
    @Published var maskUnavailable = false
    @Published var duration: Double = 0
    @Published var currentTime: Double = 0
    @Published private(set) var sourceURL: URL?
    @Published private(set) var estimatedCredits: Double = 1

    private var pollTask: Task<Void, Never>?
    private var seekTask: Task<Void, Never>?
    private var previewTask: Task<Void, Never>?
    private var nextSelectionID = 0

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

    var allPoints: [SelectionPoint] { selections.flatMap(\.points) }

    var canProcess: Bool {
        guard frame != nil else { return false }
        switch mode {
        case .moving: return allPoints.contains { $0.label == 1 }
        case .fixed: return !maskBuilder.isEmpty
        }
    }

    var activeSelection: Selection? {
        guard let activeSelectionID else { return nil }
        return selections.first { $0.id == activeSelectionID }
    }

    deinit {
        pollTask?.cancel()
        seekTask?.cancel()
        previewTask?.cancel()
    }

    // MARK: - Loading

    /// The removal editor caps clips at 90 seconds, same as the website.
    /// Background replacement is a different pipeline with its own, longer cap.
    static let maxDuration: Double = 90

    func load(videoURL: URL) async {
        reset()
        sourceURL = videoURL
        do {
            duration = await VideoFrameExtractor.duration(of: videoURL)
            currentTime = 0

            guard duration <= Self.maxDuration else {
                let seconds = Int(duration.rounded())
                AppLog.error(.editor, "Rejected \(seconds)s clip (removal limit 90s)")
                stage = .failed("That clip is \(seconds)s. Removal takes up to 90 seconds. Trim it, or use the Background tab, which allows 10 minutes.")
                return
            }


            let first = try await VideoFrameExtractor.frame(from: videoURL)
            frame = first
            maskBuilder.begin(videoSize: first.pixelSize)
            estimatedCredits = CreditEstimate.credits(
                duration: duration, size: first.pixelSize, isBackground: false
            )
            stage = .selecting
            AppLog.info(.editor, "Loaded \(Int(first.pixelSize.width))×\(Int(first.pixelSize.height)) @ \(String(format: "%.0f", first.fps))fps, \(String(format: "%.1f", duration))s")
        } catch {
            AppLog.error(.editor, "Load failed: \(error.localizedDescription)")
            stage = .failed(error.localizedDescription)
        }
    }

    func reset() {
        pollTask?.cancel(); pollTask = nil
        seekTask?.cancel(); seekTask = nil
        previewTask?.cancel(); previewTask = nil
        stage = .picking
        frame = nil
        maskBuilder.clear()
        selections = []
        activeSelectionID = nil
        nextSelectionID = 0
        maskUnavailable = false
        duration = 0
        currentTime = 0
        sourceURL = nil
    }

    /// Moves to another frame. Marks belong to the frame they were placed on,
    /// so they are dropped rather than silently applied to the wrong one.
    func seek(to seconds: Double) {
        guard let sourceURL else { return }
        seekTask?.cancel()
        seekTask = Task { [weak self] in
            guard let newFrame = try? await VideoFrameExtractor.frame(from: sourceURL, at: seconds) else { return }
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self else { return }
                self.frame = newFrame
                self.currentTime = seconds
                if !self.selections.isEmpty {
                    self.selections = []
                    self.activeSelectionID = nil
                    self.nextSelectionID = 0
                    self.maskUnavailable = false
                    AppLog.debug(.editor, "Frame \(newFrame.frameIndex): cleared previous marks")
                }
            }
        }
    }

    // MARK: - Marking

    /// `normalized` is 0…1 across the displayed frame; converting here means the
    /// view never needs to know the video's pixel size.
    func addPoint(normalized: CGPoint, label: Int) {
        guard let frame else { return }
        let x = Int((normalized.x * frame.pixelSize.width).rounded())
        let y = Int((normalized.y * frame.pixelSize.height).rounded())
        let point = SelectionPoint(x: x, y: y, label: label)

        if let id = activeSelectionID, let index = selections.firstIndex(where: { $0.id == id }) {
            selections[index].points.append(point)
        } else {
            var selection = Selection(id: nextSelectionID)
            selection.points = [point]
            selections.append(selection)
            activeSelectionID = selection.id
            nextSelectionID += 1
        }
        refreshPreview()
    }

    /// Ends the current object so the next tap starts a fresh one in a new colour.
    func startNewObject() {
        guard activeSelection?.points.isEmpty == false else { return }
        activeSelectionID = nil
    }

    func undoLastPoint() {
        guard let id = activeSelectionID ?? selections.last?.id,
              let index = selections.firstIndex(where: { $0.id == id }) else { return }

        selections[index].points.removeLast()
        if selections[index].points.isEmpty {
            selections.remove(at: index)
            activeSelectionID = selections.last?.id
            nextSelectionID = max(0, nextSelectionID - 1)
        } else {
            activeSelectionID = id
            refreshPreview()
        }
    }

    func removeSelection(_ id: Int) {
        selections.removeAll { $0.id == id }
        if activeSelectionID == id { activeSelectionID = selections.last?.id }
    }

    /// Clears every kind of mark. Used by Clear and when the mode changes:
    /// a drawn area means nothing to the tracker and points mean nothing to
    /// the static path, so carrying either across silently leaves stale work
    /// in the builder that the next Clear cannot reach.
    func clearAll() {
        previewTask?.cancel(); previewTask = nil
        selections = []
        activeSelectionID = nil
        nextSelectionID = 0
        maskUnavailable = false
        maskBuilder.clear()
    }

    /// Best-effort preview. It needs the interactive SAM2 worker, which isn't
    /// always up — a miss must never block the actual job.
    private func refreshPreview() {
        guard let frame, let id = activeSelectionID,
              let selection = selections.first(where: { $0.id == id }),
              !selection.points.isEmpty else { return }

        previewTask?.cancel()
        previewTask = Task { [weak self] in
            await MainActor.run { self?.isPreviewingMask = true }
            defer { Task { @MainActor in self?.isPreviewingMask = false } }

            // An 8K frame as base64 PNG is well over 100 MB and would stall the
            // request. The preview only has to look right, so send a capped
            // copy with the points rescaled to match; the real job further down
            // still gets full-resolution coordinates.
            guard let payload = VideoFrameExtractor.previewPayload(
                frame: frame, points: selection.points
            ) else { return }

            do {
                let mask = try await APIClient.shared.previewMask(
                    frameBase64PNG: payload.base64PNG,
                    frameIndex: frame.frameIndex,
                    points: payload.points,
                    videoWidth: Int(payload.size.width),
                    videoHeight: Int(payload.size.height)
                )
                guard !Task.isCancelled, let mask else { return }

                let tinted = MaskRenderer.tinted(
                    base64: mask, color: SelectionPalette.uiColor(selection.colorIndex)
                )
                if let tinted {
                    let m = tinted.size, f = frame.pixelSize
                    let sameAspect = abs(m.width / m.height - f.width / f.height) < 0.01
                    AppLog.info(.editor,
                        "Mask \(Int(m.width))×\(Int(m.height)) vs frame \(Int(f.width))×\(Int(f.height)) aspect \(sameAspect ? "matches" : "DIFFERS, stretching to fit")")
                }
                await MainActor.run {
                    guard let self, let index = self.selections.firstIndex(where: { $0.id == id }) else { return }
                    self.selections[index].mask = tinted
                    self.maskUnavailable = false
                }
            } catch {
                guard !Task.isCancelled else { return }
                AppLog.debug(.editor, "Mask preview unavailable: \(error.localizedDescription)")
                await MainActor.run { self?.maskUnavailable = true }
            }
        }
    }

    // MARK: - Processing

    func process(appState: AppState) async {
        guard let sourceURL, let frame else { return }
        guard appState.credits >= estimatedCredits else {
            stage = .failed(APIError.outOfCredits.localizedDescription)
            return
        }

        stage = .uploading
        AppLog.info(.editor, "Uploading \(sourceURL.lastPathComponent)")
        do {
            let taskId = try await APIClient.shared.upload(
                fileURL: sourceURL,
                filename: sourceURL.lastPathComponent,
                contentType: contentType(for: sourceURL)
            )

            let jobId: String
            switch mode {
            case .moving:
                // Full-resolution coordinates here — this is the run that matters.
                jobId = try await APIClient.shared.processVideo(
                    taskId: taskId,
                    points: allPoints,
                    videoWidth: Int(frame.pixelSize.width),
                    videoHeight: Int(frame.pixelSize.height),
                    frameIndex: frame.frameIndex
                )
                AppLog.info(.editor, "Job \(jobId): tracking \(selections.count) object(s), \(allPoints.count) point(s), frame \(frame.frameIndex)")

            case .fixed:
                guard let mask = maskBuilder.exportBase64PNG() else {
                    stage = .failed("Couldn't build the mask.")
                    return
                }
                jobId = try await APIClient.shared.processStaticMask(
                    taskId: taskId,
                    maskBase64PNG: mask,
                    videoWidth: Int(frame.pixelSize.width),
                    videoHeight: Int(frame.pixelSize.height),
                    frameCount: max(1, Int((duration * frame.fps).rounded()))
                )
                AppLog.info(.editor, "Job \(jobId): static mask over \(Int((duration * frame.fps).rounded())) frames")
            }
            JobStore.shared.record(id: jobId, kind: .removal)
            stage = .processing(progress: 0, message: "Waiting for a GPU…")
            startPolling(jobId: jobId, appState: appState)
        } catch {
            AppLog.error(.editor, "Processing failed: \(error.localizedDescription)")
            stage = .failed(error.localizedDescription)
        }
    }

    private func startPolling(jobId: String, appState: AppState) {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            // Renders run minutes, not seconds. Poll gently, and give up after
            // 20 minutes rather than spinning forever.
            let deadline = Date().addingTimeInterval(20 * 60)

            while !Task.isCancelled && Date() < deadline {
                try? await Task.sleep(for: .seconds(3))
                guard !Task.isCancelled else { return }

                guard let status = try? await APIClient.shared.jobStatus(jobId: jobId) else {
                    continue // a blip shouldn't kill the job
                }

                switch status.status {
                case "completed":
                    await self?.finish(status: status, jobId: jobId, appState: appState)
                    return
                case "failed", "error":
                    AppLog.error(.editor, "Job failed: \(status.error ?? "unknown")")
                    await MainActor.run { JobStore.shared.fail(id: jobId, reason: status.error) }
                    await MainActor.run {
                        self?.stage = .failed(status.error ?? "Processing failed on the worker.")
                    }
                    return
                default:
                    await MainActor.run {
                        self?.stage = .processing(
                            progress: status.progress ?? 0,
                            message: status.message ?? "Processing…"
                        )
                    }
                }
            }

            await MainActor.run {
                self?.stage = .failed("Timed out after 20 minutes. The job may still finish on the website.")
            }
        }
    }

    private func finish(status: JobStatusResponse, jobId: String, appState: AppState) async {
        guard let raw = status.resultURL,
              let url = APIClient.shared.absoluteResultURL(raw) else {
            stage = .failed("The job finished but returned no video.")
            return
        }

        JobStore.shared.finish(id: jobId, resultURL: raw)
        do {
            let local = try await APIClient.shared.download(url)
            AppLog.info(.editor, "Result downloaded")
            stage = .done(local)
        } catch {
            stage = .failed(error.localizedDescription)
        }
        await appState.refreshUser()
    }

    private func contentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "mov": return "video/quicktime"
        case "webm": return "video/webm"
        case "avi": return "video/x-msvideo"
        case "mkv": return "video/x-matroska"
        default: return "video/mp4"
        }
    }
}
