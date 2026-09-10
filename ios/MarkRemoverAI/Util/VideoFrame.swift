import AVFoundation
import UIKit

struct VideoFrame {
    let image: UIImage
    /// Natural pixel size with the track's rotation already applied. Points
    /// sent to SAM2 must be in this space.
    let pixelSize: CGSize
    let frameIndex: Int
    let fps: Double
}

enum VideoFrameError: LocalizedError {
    case noVideoTrack
    case extractionFailed

    var errorDescription: String? {
        switch self {
        case .noVideoTrack: return "That file doesn't contain a video track."
        case .extractionFailed: return "Couldn't read a frame from that video."
        }
    }
}

enum VideoFrameExtractor {
    /// Longest edge used for the interactive mask preview. An 8K frame encoded
    /// as base64 PNG runs to hundreds of megabytes, which no request should
    /// carry — and SAM2's preview mask only has to look right on screen.
    static let previewMaxEdge: CGFloat = 1600

    /// Grabs a frame to mark up. Zero tolerance matters: at 250fps the frames
    /// either side are 4ms away, and SAM2 seeds its track from the exact frame
    /// index we report.
    static func frame(from url: URL, at seconds: Double = 0) async throws -> VideoFrame {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoFrameError.noVideoTrack
        }

        let (naturalSize, transform, nominalFrameRate) = try await track.load(
            .naturalSize, .preferredTransform, .nominalFrameRate
        )

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let requested = CMTime(seconds: seconds, preferredTimescale: 600)
        let cgImage: CGImage
        let actualTime: CMTime
        do {
            let result = try await generator.image(at: requested)
            cgImage = result.image
            actualTime = result.actualTime
        } catch {
            throw VideoFrameError.extractionFailed
        }

        // Portrait phone video carries its rotation in the transform, so the
        // upright size is what the worker sees once it decodes.
        let oriented = naturalSize.applying(transform)
        let pixelSize = CGSize(width: abs(oriented.width), height: abs(oriented.height))

        let fps = nominalFrameRate > 0 ? Double(nominalFrameRate) : 30
        // Derive the index from the frame actually returned, not the time we
        // asked for — they differ whenever a seek lands on a keyframe boundary.
        let index = Int((actualTime.seconds * fps).rounded())

        return VideoFrame(
            image: UIImage(cgImage: cgImage),
            pixelSize: pixelSize,
            frameIndex: max(0, index),
            fps: fps
        )
    }

    struct PreviewPayload {
        let base64PNG: String
        let points: [SelectionPoint]
        let size: CGSize
    }

    /// Builds a preview-sized copy of the frame with the points rescaled to
    /// match. Anything at or under the cap is sent untouched.
    static func previewPayload(frame: VideoFrame, points: [SelectionPoint]) -> PreviewPayload? {
        let longest = max(frame.pixelSize.width, frame.pixelSize.height)

        guard longest > previewMaxEdge else {
            guard let data = frame.image.pngData() else { return nil }
            return PreviewPayload(
                base64PNG: data.base64EncodedString(),
                points: points,
                size: frame.pixelSize
            )
        }

        let scale = previewMaxEdge / longest
        let target = CGSize(
            width: (frame.pixelSize.width * scale).rounded(),
            height: (frame.pixelSize.height * scale).rounded()
        )

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let scaled = UIGraphicsImageRenderer(size: target, format: format).image { _ in
            frame.image.draw(in: CGRect(origin: .zero, size: target))
        }

        guard let data = scaled.pngData() else { return nil }

        let moved = points.map {
            // objectId has to come along: dropping it here let every click on a
            // 4K clip collapse into object 0, since only downscaled frames pass
            // through this branch.
            SelectionPoint(
                x: Int((Double($0.x) * scale).rounded()),
                y: Int((Double($0.y) * scale).rounded()),
                label: $0.label,
                objectId: $0.objectId
            )
        }

        AppLog.debug(.editor, "Preview downscaled \(Int(frame.pixelSize.width))→\(Int(target.width))px for the mask request")
        return PreviewPayload(base64PNG: data.base64EncodedString(), points: moved, size: target)
    }

    static func duration(of url: URL) async -> Double {
        let asset = AVURLAsset(url: url)
        guard let seconds = try? await asset.load(.duration).seconds, seconds.isFinite else {
            return 0
        }
        return seconds
    }
}
