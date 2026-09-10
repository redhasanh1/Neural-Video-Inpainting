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

    @Published var stage: Stage = .picking
    @Published var frame: VideoFrame?
    @Published var points: [SelectionPoint] = []
    @Published var maskImage: UIImage?
    @Published var isPreviewingMask = false
    @Published var maskUnavailable = false

    private var sourceURL: URL?
    private var pollTask: Task<Void, Never>?

    var canProcess: Bool {
        frame != nil && points.contains { $0.label == 1 }
    }

    deinit { pollTask?.cancel() }

    // MARK: - Picking

    func load(videoURL: URL) async {
        reset()
        sourceURL = videoURL
        do {
            frame = try await VideoFrameExtractor.frame(from: videoURL)
            stage = .selecting
        } catch {
            stage = .failed(error.localizedDescription)
        }
    }

    func reset() {
        pollTask?.cancel()
        pollTask = nil
        stage = .picking
        frame = nil
        points = []
        maskImage = nil
        maskUnavailable = false
        sourceURL = nil
    }

    // MARK: - Points

    /// `normalized` is the tap expressed 0…1 across the displayed frame, so the
    /// caller doesn't need to know the video's pixel size.
    func addPoint(normalized: CGPoint, label: Int) {
        guard let frame else { return }
        let x = Int((normalized.x * frame.pixelSize.width).rounded())
        let y = Int((normalized.y * frame.pixelSize.height).rounded())
        points.append(SelectionPoint(x: x, y: y, label: label))
        Task { await refreshMaskPreview() }
    }

    func undoLastPoint() {
        guard !points.isEmpty else { return }
        points.removeLast()
        if points.isEmpty {
            maskImage = nil
        } else {
            Task { await refreshMaskPreview() }
        }
    }

    func clearPoints() {
        points = []
        maskImage = nil
        maskUnavailable = false
    }

    /// Best-effort: the preview needs the interactive SAM2 worker, which isn't
    /// always up. A miss shouldn't block the actual job.
    private func refreshMaskPreview() async {
        guard let frame, !points.isEmpty else { return }
        guard let base64 = VideoFrameExtractor.base64PNG(frame.image) else { return }

        isPreviewingMask = true
        defer { isPreviewingMask = false }

        do {
            let mask = try await APIClient.shared.previewMask(
                frameBase64PNG: base64,
                frameIndex: frame.frameIndex,
                points: points,
                videoWidth: Int(frame.pixelSize.width),
                videoHeight: Int(frame.pixelSize.height)
            )
            if let mask, let image = VideoFrameExtractor.maskImage(fromBase64: mask) {
                maskImage = image
                maskUnavailable = false
            }
        } catch {
            maskUnavailable = true
        }
    }

    // MARK: - Processing

    func process(appState: AppState) async {
        guard let sourceURL, let frame else { return }
        guard appState.credits >= 1 else {
            stage = .failed(APIError.outOfCredits.localizedDescription)
            return
        }

        stage = .uploading
        do {
            let taskId = try await APIClient.shared.upload(
                fileURL: sourceURL,
                filename: sourceURL.lastPathComponent,
                contentType: contentType(for: sourceURL)
            )

            let jobId = try await APIClient.shared.processVideo(
                taskId: taskId,
                points: points,
                videoWidth: Int(frame.pixelSize.width),
                videoHeight: Int(frame.pixelSize.height),
                frameIndex: frame.frameIndex
            )

            stage = .processing(progress: 0, message: "Waiting for a GPU…")
            startPolling(jobId: jobId, appState: appState)
        } catch {
            stage = .failed(error.localizedDescription)
        }
    }

    private func startPolling(jobId: String, appState: AppState) {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            // Renders run minutes, not seconds — poll gently and give up after
            // 20 minutes rather than spinning forever.
            let deadline = Date().addingTimeInterval(20 * 60)

            while !Task.isCancelled && Date() < deadline {
                try? await Task.sleep(for: .seconds(3))
                guard !Task.isCancelled else { return }

                guard let status = try? await APIClient.shared.jobStatus(jobId: jobId) else {
                    continue // a blip in the tunnel shouldn't kill the job
                }

                switch status.status {
                case "completed":
                    await self?.finish(status: status, appState: appState)
                    return
                case "failed", "error":
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

    private func finish(status: JobStatusResponse, appState: AppState) async {
        guard let raw = status.resultURL,
              let url = APIClient.shared.absoluteResultURL(raw) else {
            stage = .failed("The job finished but returned no video.")
            return
        }

        do {
            let local = try await APIClient.shared.download(url)
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
