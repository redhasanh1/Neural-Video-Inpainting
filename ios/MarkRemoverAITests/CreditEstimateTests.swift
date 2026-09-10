import XCTest
@testable import MarkRemoverAI

/// The number quoted in the app has to match what the server charges, or the
/// balance moves by a different amount than the paywall promised.
final class CreditEstimateTests: XCTestCase {

    private let hd = CGSize(width: 1280, height: 720)
    private let uhd = CGSize(width: 3840, height: 2160)

    func testRemovalIsAlwaysOneCredit() {
        XCTAssertEqual(CreditEstimate.credits(duration: 1, size: hd, isBackground: false), 1)
        XCTAssertEqual(CreditEstimate.credits(duration: 90, size: uhd, isBackground: false), 1)
        // Even nonsense input still costs the flat rate, matching the website.
        XCTAssertEqual(CreditEstimate.credits(duration: 0, size: .zero, isBackground: false), 1)
    }

    func testBackgroundAtBaselineIsAThirdOfOne() {
        // 10s of 720p is the baseline: both factors are 1, so raw is 1 and the
        // quote is 1/3 rounded to one decimal.
        XCTAssertEqual(CreditEstimate.credits(duration: 10, size: hd, isBackground: true), 0.3, accuracy: 0.0001)
    }

    func testBackgroundScalesWithDurationAndResolution() {
        let short = CreditEstimate.credits(duration: 10, size: hd, isBackground: true)
        let long = CreditEstimate.credits(duration: 60, size: hd, isBackground: true)
        let big = CreditEstimate.credits(duration: 10, size: uhd, isBackground: true)

        XCTAssertGreaterThan(long, short)
        XCTAssertGreaterThan(big, short)
    }

    func testBackgroundNeverQuotesZero() {
        XCTAssertEqual(CreditEstimate.credits(duration: 0, size: hd, isBackground: true), 0.1)
        XCTAssertEqual(CreditEstimate.credits(duration: 10, size: .zero, isBackground: true), 0.1)
        // A clip short enough to round to zero still has to cost something.
        XCTAssertGreaterThanOrEqual(CreditEstimate.credits(duration: 0.01, size: CGSize(width: 2, height: 2), isBackground: true), 0.1)
    }

    func testLabelDropsTheDecimalOnWholeNumbers() {
        XCTAssertEqual(CreditEstimate.label(3), "3")
        XCTAssertEqual(CreditEstimate.label(0.3), "0.3")
        XCTAssertEqual(CreditEstimate.label(12.5), "12.5")
    }

    func testCompactAbbreviatesLargeBalances() {
        XCTAssertEqual(CreditEstimate.compact(5), "5")
        XCTAssertEqual(CreditEstimate.compact(999), "999")
        XCTAssertEqual(CreditEstimate.compact(1_500), "1.5K")
        XCTAssertEqual(CreditEstimate.compact(12_000), "12K")
        XCTAssertEqual(CreditEstimate.compact(2_400_000), "2.4M")
    }

    func testCompactPromotesInsteadOfPrinting1000() {
        // 999,999,604 / 1_000_000 formats as "1000M", which is not a number
        // anyone writes — it has to promote to the next tier instead.
        XCTAssertEqual(CreditEstimate.compact(999_999_604), "1.0B")
        XCTAssertEqual(CreditEstimate.compact(999_600), "1.0M")
        XCTAssertEqual(CreditEstimate.compact(999.6), "1.0K")
    }

    // MARK: - Time estimate

    func testBaselineClipQuotesThePaddedMinute() {
        // 10s at 720p is the baseline: 60s of work, padded by 1.25.
        let s = CreditEstimate.seconds(duration: 10, size: CGSize(width: 1280, height: 720),
                                       isBackground: false)
        XCTAssertEqual(s, 75, accuracy: 0.01)
    }

    func testPaddingIsAppliedNotIgnored() {
        let padded = CreditEstimate.seconds(duration: 10, size: CGSize(width: 1280, height: 720),
                                            isBackground: false)
        XCTAssertEqual(padded / CreditEstimate.padding, 60, accuracy: 0.01)
    }

    func testLongerClipsTakeLongerEvenThoughRemovalIsFlatPriced() {
        let short = CreditEstimate.seconds(duration: 10, size: CGSize(width: 1280, height: 720),
                                           isBackground: false)
        let long = CreditEstimate.seconds(duration: 90, size: CGSize(width: 1280, height: 720),
                                          isBackground: false)
        XCTAssertGreaterThan(long, short * 4)
        // ...while the price stays flat, which is the mismatch being explained.
        XCTAssertEqual(CreditEstimate.credits(duration: 10, size: CGSize(width: 1280, height: 720),
                                              isBackground: false),
                       CreditEstimate.credits(duration: 90, size: CGSize(width: 1280, height: 720),
                                              isBackground: false))
    }

    func testHigherResolutionTakesLonger() {
        let hd = CreditEstimate.seconds(duration: 10, size: CGSize(width: 1280, height: 720),
                                        isBackground: false)
        let uhd = CreditEstimate.seconds(duration: 10, size: CGSize(width: 3840, height: 2160),
                                         isBackground: false)
        XCTAssertGreaterThan(uhd, hd)
    }

    func testBackgroundIsQuotedLighterThanRemoval() {
        let size = CGSize(width: 1280, height: 720)
        XCTAssertLessThan(CreditEstimate.seconds(duration: 10, size: size, isBackground: true),
                          CreditEstimate.seconds(duration: 10, size: size, isBackground: false))
    }

    func testNeverQuotesLessThanFifteenSeconds() {
        let s = CreditEstimate.seconds(duration: 0.5, size: CGSize(width: 320, height: 240),
                                       isBackground: true)
        XCTAssertGreaterThanOrEqual(s, 15)
    }

    func testZeroSizedClipReturnsNoEstimate() {
        XCTAssertEqual(CreditEstimate.seconds(duration: 0, size: .zero, isBackground: false), 0)
    }

    func testTimeLabelReadsNaturally() {
        XCTAssertEqual(CreditEstimate.timeLabel(45), "45s")
        XCTAssertEqual(CreditEstimate.timeLabel(75), "1 min")
        XCTAssertEqual(CreditEstimate.timeLabel(150), "3 min")
        XCTAssertEqual(CreditEstimate.timeLabel(3600), "1h")
        XCTAssertEqual(CreditEstimate.timeLabel(3900), "1h 5m")
    }

    func testWaitLabelFallsBackWhenSizeUnknown() {
        XCTAssertTrue(CreditEstimate.waitLabel(duration: 0, size: .zero, isBackground: false)
            .contains("minute or two"))
    }
}
