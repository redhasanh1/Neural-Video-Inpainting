# ObjectRemoverAI — App Store submission pack

Apple ID `6804926856` · bundle `com.markremoverai.app` · team `KD3JCFB7N4`

RoomFinderAI was rejected under **Guideline 2.1 — Information Needed**. Apple
asked six questions. Everything below is written to answer them up front so the
same round trip does not happen twice.

---

## App Review Information → Notes

Copy this into the Notes field.

> ObjectRemoverAI removes unwanted objects from video. The user picks a clip
> from their library, marks the object, and our GPU service rebuilds the
> background behind it frame by frame. A second tool replaces the background
> behind a subject instead.
>
> **Test account**
> Email: <FILL IN>
> Password: <FILL IN>
> This account has credits already applied, so no purchase is needed to
> exercise the full flow.
>
> **How to exercise the app**
> 1. Sign in with the account above (Sign in with Apple and Google also work).
> 2. Objects tab → Choose a video → pick any clip under 90 seconds.
> 3. Tap the object to remove. A coloured mask appears over it.
> 4. Tap "Remove it". Rendering takes roughly one minute for a ten second clip.
> 5. The result opens on a before/after wipe, and can be saved to Photos.
> 6. Background tab does the same but replaces the background. Limit is ten
>    minutes there.
> 7. Renders continue if the app is closed. Profile → Renders collects them.
> 8. Account deletion: Profile → Delete account. It asks for a typed
>    confirmation and removes the account, uploads and results.
>
> **Devices tested**
> iPhone 17 simulator on iOS 26.5, and <FILL IN physical device + iOS version>.
>
> **What the app is for**
> People who have footage with something in it they did not want: a passer-by,
> a bin, a sign. Editing that out by hand is slow and needs skill. The app does
> it from a single tap.
>
> **External services used**
> - Railway hosts our API.
> - Backblaze B2 stores the uploaded clip and the result.
> - Salad Cloud provides the GPU workers that run the models.
> - Models: SAM 2 for tracking the selection, ProPainter for filling in behind it.
> - Sign in with Apple and Google Sign-In for authentication.
> - Apple In-App Purchase for credits. No other payment processor is used in
>   the app.
>
> **Regional differences**
> None. The app behaves identically in every region.
>
> **Content rights**
> The app processes only video the user supplies from their own library. It
> hosts no catalogue, no user-to-user sharing, and no third-party content.

---

## App Store listing

**Name** ObjectRemoverAI
**Subtitle** Erase objects from your videos

**Promotional text**
> Point at anything you want gone. The background behind it is rebuilt frame by
> frame, at your original resolution.

**Description**
> Some footage has something in it you did not want. A passer-by walking
> through the shot. A bin at the edge of frame. A sign you cannot crop out
> without losing the composition.
>
> ObjectRemoverAI takes it out.
>
> Pick a clip, tap what should go, and the object is tracked across every frame
> and removed. What was behind it is reconstructed, so the result looks like it
> was never there. Your original resolution is preserved, up to 4K and beyond.
>
> TWO TOOLS
> Objects — remove something from the shot. Tap a moving subject to have it
> followed, or draw over anything that stays in one place.
> Background — keep the subject and replace everything behind them with
> transparency, a blur, or a solid colour.
>
> BUILT FOR PRECISION
> Pinch to zoom before you tap, so a small detail in a corner is easy to hit.
> Add or subtract from a selection until it is right. Adjust how far the mask
> spreads to avoid a halo. Preview the mask before you spend anything.
>
> HONEST ABOUT COST
> The price of a render is shown before you start it, never after.
>
> Renders keep going if you close the app, and are waiting for you when you
> come back.

**Keywords**
> video editor,remove object,erase,cleanup,background,retouch,inpaint,video

**Support URL** https://markremoverai.com/contact
**Marketing URL** https://markremoverai.com
**Privacy Policy URL** https://markremoverai.com/privacy

---

## In-app purchases

| Product ID | Reference | Display name | Price |
|---|---|---|---|
| `com.markremoverai.app.credits5` | Starter Pack | 5 Credits | $2.99 |
| `com.markremoverai.app.credits15` | Basic Pack | 15 Credits | $6.99 |
| `com.markremoverai.app.credits60` | Pro Pack | 60 Credits | $24.99 |

Description for each: "Credits for removing objects from your videos. One
credit removes an object from one video. Credits never expire."

**Blocked until:** Paid Apps Agreement leaves "Pending User Info". That needs a
bank account and the two tax forms (Canadian GST/HST 506 and the U.S. tax
questionnaire) under Business.

---

## Age rating

4+. No objectionable content. The app processes only what the user supplies.

## App Privacy

- **Email address** — linked to the user, used for account management only.
- **User content (video)** — uploaded to process, not linked to identity for
  tracking, not used for advertising. Deleted with the account.
- No tracking, no third-party advertising, no data brokers.

---

## Submitted

**iOS 1.0 — Waiting for Review.** Four items went in together on 25 August 2026:
the app version and all three credit packs, which is what Apple requires for a
first consumable.

- **Build** 1.0 (2), signed with the team distribution profile,
  `ITSAppUsesNonExemptEncryption` false so export compliance is answered in the
  binary, iPhone only.
- **Icon** the wand mark from markremoverai.com, 1024x1024, no alpha.
- **Screenshots** five at 1284x2778 (iPhone 6.5"): Objects home, the editor with
  a live mask on a real clip, Guide, Background, Profile.
- **App Privacy** published. Five data types (Name, Email Address, Photos or
  Videos, User ID, Purchase History), each App Functionality, each linked to the
  user, none used for tracking.
- **Age rating** 4+, no override. **Pricing** free in 175 countries or regions.
- **In-app purchases** 5 Credits $2.99, 15 Credits $6.99, 60 Credits $24.99,
  each with price, availability, localisation, review screenshot and notes.
- **Tests** 31 unit tests over the credit estimate, the API decoding, the mask
  builder and the job store. All passing.

### Why no demo account

The reviewer notes tell Apple to use Sign in with Apple. That path creates the
account immediately, skips email verification, and grants 2 free credits, which
is enough for two complete removals. `demoAccountRequired` is false for that
reason. If Apple asks for one anyway, add it under App Review Information.

### Still outstanding

1. **Bank account and the two tax forms** (Canadian GST/HST 506 and the U.S. tax
   questionnaire) under Business. The app can ship without them; the credit
   packs cannot go on sale until the Paid Apps agreement is active.
2. **Better review screenshots for the three packs.** Each carries one, but it
   shows the paywall's empty state. StoreKit only populates the packs against
   `Products.storekit` when the app is launched from Xcode, and Apple's sandbox
   returns an empty product list until the purchases are approved.
3. **The website's demo videos are Sora clips.** `web/demos/before.mp4` carries a
   visible Sora watermark and both are a Mister Rogers likeness. They are live on
   markremoverai.com and should be replaced.
