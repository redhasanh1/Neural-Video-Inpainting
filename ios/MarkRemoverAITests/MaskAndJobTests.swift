import XCTest
import UIKit
@testable import MarkRemoverAI

/// The mask pipeline is the part users see go wrong first: a mask that lands in
/// the wrong place, or a brush that smears instead of previewing.
final class StaticMaskBuilderTests: XCTestCase {

    private func nonBlackFraction(_ base64: String) throws -> Double {
        let data = try XCTUnwrap(Data(base64Encoded: base64))
        let image = try XCTUnwrap(UIImage(data: data)?.cgImage)

        let width = image.width, height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height)
        let context = try XCTUnwrap(CGContext(
            data: &pixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ))
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return Double(pixels.filter { $0 > 127 }.count) / Double(width * height)
    }

    @MainActor func testAFreshMaskIsEntirelyBlack() throws {
        let builder = StaticMaskBuilder()
        builder.begin(videoSize: CGSize(width: 640, height: 360))

        let painted = try nonBlackFraction(try XCTUnwrap(builder.exportBase64PNG()))
        XCTAssertEqual(painted, 0, accuracy: 0.001)
    }

    @MainActor func testBrushStrokePaintsWhite() throws {
        let builder = StaticMaskBuilder()
        builder.begin(videoSize: CGSize(width: 640, height: 360))
        builder.stroke(from: CGPoint(x: 0.2, y: 0.5), to: CGPoint(x: 0.8, y: 0.5), tool: .brush, brushFraction: 0.1)
        builder.endStroke()

        XCTAssertGreaterThan(try nonBlackFraction(try XCTUnwrap(builder.exportBase64PNG())), 0.01)
    }

    @MainActor func testClearWipesThePaintedMask() throws {
        let builder = StaticMaskBuilder()
        builder.begin(videoSize: CGSize(width: 640, height: 360))
        builder.stroke(from: CGPoint(x: 0.1, y: 0.5), to: CGPoint(x: 0.9, y: 0.5), tool: .brush, brushFraction: 0.2)
        builder.endStroke()
        XCTAssertGreaterThan(try nonBlackFraction(try XCTUnwrap(builder.exportBase64PNG())), 0)

        builder.clear()
        XCTAssertEqual(try nonBlackFraction(try XCTUnwrap(builder.exportBase64PNG())), 0, accuracy: 0.001)
    }

    @MainActor func testEraserRemovesWhatTheBrushPainted() throws {
        let builder = StaticMaskBuilder()
        builder.begin(videoSize: CGSize(width: 640, height: 360))
        builder.stroke(from: CGPoint(x: 0.1, y: 0.5), to: CGPoint(x: 0.9, y: 0.5), tool: .brush, brushFraction: 0.2)
        builder.endStroke()
        let painted = try nonBlackFraction(try XCTUnwrap(builder.exportBase64PNG()))

        builder.stroke(from: CGPoint(x: 0.1, y: 0.5), to: CGPoint(x: 0.9, y: 0.5), tool: .eraser, brushFraction: 0.3)
        builder.endStroke()

        XCTAssertLessThan(try nonBlackFraction(try XCTUnwrap(builder.exportBase64PNG())), painted)
    }

    @MainActor func testDraggingARectanglePreviewsRatherThanAccumulating() throws {
        // Every drag event redraws from a snapshot. Without that, shrinking the
        // box left the larger one behind and the tool scribbled.
        let builder = StaticMaskBuilder()
        builder.begin(videoSize: CGSize(width: 640, height: 360))

        builder.stroke(from: CGPoint(x: 0.1, y: 0.1), to: CGPoint(x: 0.9, y: 0.9), tool: .rectangle, brushFraction: 0.05)
        let large = try nonBlackFraction(try XCTUnwrap(builder.exportBase64PNG()))

        builder.stroke(from: CGPoint(x: 0.1, y: 0.1), to: CGPoint(x: 0.3, y: 0.3), tool: .rectangle, brushFraction: 0.05)
        let small = try nonBlackFraction(try XCTUnwrap(builder.exportBase64PNG()))
        builder.endStroke()

        XCTAssertLessThan(small, large / 2, "the earlier, larger box was left behind")
    }

    @MainActor func testExportMatchesTheVideoAspectRatio() throws {
        let builder = StaticMaskBuilder()
        builder.begin(videoSize: CGSize(width: 704, height: 1280))

        let data = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(builder.exportBase64PNG())))
        let image = try XCTUnwrap(UIImage(data: data)?.cgImage)

        XCTAssertEqual(Double(image.width) / Double(image.height), 704.0 / 1280.0, accuracy: 0.01)
    }

    @MainActor func testAnEightKFrameIsCappedSoTheBufferStaysAffordable() {
        // 8K is 33M pixels. Past 4K the extra precision buys nothing for a
        // region mask and just costs memory.
        let builder = StaticMaskBuilder()
        builder.begin(videoSize: CGSize(width: 7680, height: 4320))

        XCTAssertLessThanOrEqual(max(builder.size.width, builder.size.height), 4096)
        // The aspect still has to survive the cap, or the mask lands skewed.
        XCTAssertEqual(builder.size.width / builder.size.height, 7680.0 / 4320.0, accuracy: 0.01)
    }

    @MainActor func testASmallFrameIsNotUpscaled() {
        let builder = StaticMaskBuilder()
        builder.begin(videoSize: CGSize(width: 640, height: 360))
        XCTAssertEqual(builder.size, CGSize(width: 640, height: 360))
    }

    @MainActor func testTheMaskStartsEmptyAndStopsBeingEmptyOnceDrawn() {
        let builder = StaticMaskBuilder()
        builder.begin(videoSize: CGSize(width: 640, height: 360))
        XCTAssertTrue(builder.isEmpty)

        builder.stroke(from: CGPoint(x: 0.2, y: 0.2), to: CGPoint(x: 0.6, y: 0.6), tool: .rectangle, brushFraction: 0.05)
        builder.endStroke()
        XCTAssertFalse(builder.isEmpty)

        builder.clear()
        XCTAssertTrue(builder.isEmpty)
    }

}

final class TrackedJobTests: XCTestCase {

    func testAgeReadsInTheLargestUnitThatFits() {
        func age(_ secondsAgo: TimeInterval) -> String {
            var job = TrackedJob(id: "j", kind: .removal)
            job.submitted = Date().addingTimeInterval(-secondsAgo)
            return job.age
        }

        XCTAssertEqual(age(5), "5s ago")
        XCTAssertEqual(age(120), "2m ago")
        XCTAssertEqual(age(7200), "2h ago")
        XCTAssertEqual(age(172_800), "2d ago")
    }

    func testAJobRoundTripsThroughJSON() throws {
        var job = TrackedJob(id: "abc", kind: .background, detail: "beach")
        job.state = .finished
        job.resultURL = "https://cdn/out.mp4"

        let decoded = try JSONDecoder().decode(
            TrackedJob.self, from: try JSONEncoder().encode(job)
        )
        XCTAssertEqual(decoded, job)
    }

    func testKindCarriesItsOwnLabelAndIcon() {
        XCTAssertEqual(TrackedJob.Kind.removal.title, "Object removal")
        XCTAssertEqual(TrackedJob.Kind.background.title, "Background")
        XCTAssertFalse(TrackedJob.Kind.removal.symbol.isEmpty)
        XCTAssertFalse(TrackedJob.Kind.background.symbol.isEmpty)
    }
}
