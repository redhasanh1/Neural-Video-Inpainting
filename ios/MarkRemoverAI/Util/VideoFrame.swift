import AVFoundation
import UIKit

struct VideoFrame {
    let image: UIImage
    /// Natural pixel size of the video, orientation already applied. Point
    /// coordinates sent to SAM2 must be in this space.
    let pixelSize: CGSize
    let frameIndex: Int
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
    /// Grabs a frame for tapping on. Defaults to the first frame, which is what
    /// the backend treats as frame 0 when it seeds the SAM2 track.
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

        let time = CMTime(seconds: seconds, preferredTimescale: 600)
        let cgImage: CGImage
        do {
            cgImage = try await generator.image(at: time).image
        } catch {
            throw VideoFrameError.extractionFailed
        }

        // Portrait iPhone video carries its rotation in the transform, so the
        // upright size is what the worker sees after it decodes.
        let orientedSize = naturalSize.applying(transform)
        let pixelSize = CGSize(width: abs(orientedSize.width), height: abs(orientedSize.height))

        let fps = nominalFrameRate > 0 ? Double(nominalFrameRate) : 30
        return VideoFrame(
            image: UIImage(cgImage: cgImage),
            pixelSize: pixelSize,
            frameIndex: Int((seconds * fps).rounded())
        )
    }

    /// PNG, base64, no data: prefix — the shape `/api/sam2/select-object` wants.
    static func base64PNG(_ image: UIImage) -> String? {
        image.pngData()?.base64EncodedString()
    }

    /// Decodes the base64 PNG mask the worker returns so it can be laid over
    /// the frame.
    static func maskImage(fromBase64 base64: String) -> UIImage? {
        guard let data = Data(base64Encoded: base64) else { return nil }
        return UIImage(data: data)
    }

    static func duration(of url: URL) async -> Double {
        let asset = AVURLAsset(url: url)
        guard let seconds = try? await asset.load(.duration).seconds, seconds.isFinite else {
            return 0
        }
        return seconds
    }
}
