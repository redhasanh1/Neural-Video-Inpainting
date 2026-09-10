import Photos

enum PhotoSaverError: LocalizedError {
    case denied

    var errorDescription: String? {
        "MarkRemoverAI needs permission to add videos to your library. Enable it in Settings."
    }
}

enum PhotoSaver {
    static func saveVideo(at url: URL) async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else { throw PhotoSaverError.denied }

        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
        }
    }
}
