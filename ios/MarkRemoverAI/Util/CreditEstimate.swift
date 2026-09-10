import Foundation

/// Mirrors the website's estimate so the number quoted in the app matches what
/// the job actually costs. Baseline is 720p for 10 seconds.
enum CreditEstimate {
    static let basePixels: Double = 1280 * 720
    static let baseDuration: Double = 10

    /// The two tools price differently, and not symmetrically:
    ///
    /// - Object removal is a **flat 1 credit** whatever the clip. The website
    ///   computes the duration and resolution factors there and then discards
    ///   them — the formula is dead code on that page.
    /// - Background replacement genuinely scales, at a third of the rate.
    ///
    /// Applying the scaled formula to both quoted several credits for a long
    /// 4K removal the site charges 1 for.
    static func credits(duration: Double, size: CGSize, isBackground: Bool) -> Double {
        guard isBackground else { return 1 }
        guard duration > 0, size.width > 0, size.height > 0 else { return 0.1 }

        let durationFactor = duration / baseDuration
        let resolutionFactor = ((size.width * size.height) / basePixels).squareRoot()
        let raw = durationFactor * 0.7 + resolutionFactor * 0.3

        return max(0.1, ((raw / 3) * 10).rounded() / 10)
    }

    static func label(_ credits: Double) -> String {
        credits == credits.rounded()
            ? String(Int(credits))
            : String(format: "%.1f", credits)
    }
}


extension CreditEstimate {
    /// Balances can run to nine digits, which pushes the title out of the bar.
    /// Abbreviate anything over a thousand.
    static func compact(_ credits: Double) -> String {
        // Promote before formatting: 999,999,604 divided by a million is
        // 999.99, which prints as "1000M", which is not a number anyone writes.
        // Comparing against the tier boundary minus half a unit catches that.
        if credits >= 999_500_000 { return trim(credits / 1_000_000_000, "B") }
        if credits >= 999_500     { return trim(credits / 1_000_000, "M") }
        if credits >= 999.5       { return trim(credits / 1_000, "K") }
        return label(credits)
    }

    private static func trim(_ value: Double, _ suffix: String) -> String {
        // One decimal below ten, none above: 1.5K, 12K, 999K.
        value < 10
            ? String(format: "%.1f%@", value, suffix)
            : String(format: "%.0f%@", value, suffix)
    }
}


extension CreditEstimate {
    /// Quoted times are padded by this much. An estimate people routinely blow
    /// past reads as a hang - the render that takes three minutes against a
    /// promise of "a minute or two" looks broken even though it is fine. Erring
    /// high costs nothing; erring low costs trust.
    static let padding: Double = 1.25

    /// Roughly how long a render will take, in seconds, already padded.
    ///
    /// Same duration/resolution blend the price uses, so the two numbers move
    /// together. Note that removal is a flat *price* but not a flat *time* - a
    /// 90 second 4K clip costs the same as a 5 second one and takes far longer,
    /// which is precisely the mismatch that made the wait feel broken.
    static func seconds(duration: Double, size: CGSize, isBackground: Bool) -> Double {
        guard duration > 0, size.width > 0, size.height > 0 else { return 0 }

        let durationFactor = duration / baseDuration
        let resolutionFactor = ((size.width * size.height) / basePixels).squareRoot()
        let work = durationFactor * 0.7 + resolutionFactor * 0.3

        // Calibrated to the baseline everyone quotes: about a minute for ten
        // seconds of 720p. Background replacement is the lighter pipeline.
        let base: Double = isBackground ? 30 : 60

        return max(15, base * work * padding)
    }

    /// A short human label for a duration in seconds: "45s", "2 min", "1h 5m".
    static func timeLabel(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        if total < 60 { return "\(total)s" }
        if total < 3600 {
            let m = Int((Double(total) / 60).rounded())
            return "\(m) min"
        }
        let h = total / 3600, m = (total % 3600) / 60
        return m == 0 ? "\(h)h" : "\(h)h \(m)m"
    }

    /// The same estimate phrased for the progress screen, where it is a promise
    /// about the future rather than a column in a table.
    static func waitLabel(duration: Double, size: CGSize, isBackground: Bool) -> String {
        let s = seconds(duration: duration, size: size, isBackground: isBackground)
        guard s > 0 else { return "This usually takes a minute or two." }
        return "Usually about \(timeLabel(s)) for a clip this size."
    }
}
