import XCTest
@testable import MarkRemoverAI

/// The Flask API is loosely typed on the wire. These pin the shapes that have
/// already broken the app once.
final class APIModelDecodingTests: XCTestCase {

    private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    func testCreditsDecodeFromAJSONNumber() throws {
        let user = try decode(User.self, #"{"id":1,"email":"a@b.c","name":"A","credits":12.5,"email_verified":true}"#)
        XCTAssertEqual(user.credits, 12.5)
        XCTAssertTrue(user.emailVerified)
    }

    func testCreditsDecodeFromAJSONString() throws {
        // Postgres NUMERIC reaches Flask as a Decimal and serialises as a
        // string. Decoding only as Double silently showed a balance of zero.
        let user = try decode(User.self, #"{"id":1,"email":"a@b.c","name":null,"credits":"12.0","email_verified":false}"#)
        XCTAssertEqual(user.credits, 12)
        XCTAssertNil(user.name)
    }

    func testCreditsFallBackToZeroWhenUnparseable() throws {
        let user = try decode(User.self, #"{"id":1,"email":"a@b.c","credits":null,"email_verified":true}"#)
        XCTAssertEqual(user.credits, 0)
    }

    func testEmailVerifiedDefaultsToFalseWhenAbsent() throws {
        let user = try decode(User.self, #"{"id":1,"email":"a@b.c","credits":1}"#)
        XCTAssertFalse(user.emailVerified)
    }

    func testWithCreditsKeepsEverythingElse() {
        let user = User(id: 7, email: "a@b.c", name: "A", credits: 1, emailVerified: true)
        let updated = user.withCredits(9)
        XCTAssertEqual(updated.credits, 9)
        XCTAssertEqual(updated.id, 7)
        XCTAssertEqual(updated.email, "a@b.c")
        XCTAssertEqual(updated.name, "A")
        XCTAssertTrue(updated.emailVerified)
    }

    func testSelectionPointEncodesObjectIdUnderTheServersKey() throws {
        let point = SelectionPoint(x: 10, y: 20, label: 1, objectId: 3)
        let object = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(point)
        ) as? [String: Any]

        XCTAssertEqual(object?["object_id"] as? Int, 3)
        XCTAssertEqual(object?["x"] as? Int, 10)
        XCTAssertEqual(object?["label"] as? Int, 1)
    }

    func testSelectionPointPayloadMatchesItsEncodedForm() {
        let payload = SelectionPoint(x: 4, y: 5, label: 0, objectId: 2).payload
        XCTAssertEqual(payload["object_id"] as? Int, 2)
        XCTAssertEqual(payload["x"] as? Int, 4)
        XCTAssertEqual(payload["y"] as? Int, 5)
        XCTAssertEqual(payload["label"] as? Int, 0)
    }

    func testErrorResponseFallsBackFromErrorToMessage() throws {
        let onlyMessage = try decode(APIErrorResponse.self, #"{"message":"nope"}"#)
        XCTAssertEqual(onlyMessage.text, "nope")

        let both = try decode(APIErrorResponse.self, #"{"error":"bad","message":"nope"}"#)
        XCTAssertEqual(both.text, "bad")

        let neither = try decode(APIErrorResponse.self, "{}")
        XCTAssertEqual(neither.text, "Something went wrong.")
    }

    func testUnverifiedLoginCarriesTheFlag() throws {
        let response = try decode(APIErrorResponse.self, #"{"error":"verify","needs_verification":true}"#)
        XCTAssertEqual(response.needsVerification, true)
    }

    func testJobStatusMapsSnakeCaseKeys() throws {
        let job = try decode(JobStatusResponse.self, #"{"status":"completed","progress":100,"result_url":"https://x/y.mp4","new_credits":4}"#)
        XCTAssertEqual(job.resultURL, "https://x/y.mp4")
        XCTAssertEqual(job.newCredits, 4)
        XCTAssertEqual(job.progress, 100)
    }

    func testUploadURLResponseMapsSnakeCaseKeys() throws {
        let upload = try decode(UploadURLResponse.self, #"{"status":"ok","task_id":"t1","upload_url":"https://u","auth_token":"tok","remote_path":"p/v.mp4","cdn_url":"https://cdn/v.mp4"}"#)
        XCTAssertEqual(upload.taskId, "t1")
        XCTAssertEqual(upload.uploadURL, "https://u")
        XCTAssertEqual(upload.authToken, "tok")
        XCTAssertEqual(upload.remotePath, "p/v.mp4")
        XCTAssertEqual(upload.cdnURL, "https://cdn/v.mp4")
    }

    // MARK: - Job status

    /// The exact payload a finished render returns. `new_credits` is a string
    /// because it is a Postgres NUMERIC; decoding it as Double used to throw
    /// and strand finished jobs on the "still running" list.
    func testCompletedJobWithStringCreditsStillDecodes() throws {
        let json = #"{"new_credits":"1.00","result_url":"https://cdn/x.mp4","status":"completed"}"#
        let s = try decode(JobStatusResponse.self, json)
        XCTAssertEqual(s.status, "completed")
        XCTAssertEqual(s.resultURL, "https://cdn/x.mp4")
        XCTAssertEqual(s.newCredits, 1.0)
    }

    func testCompletedJobWithNumericCreditsStillDecodes() throws {
        let s = try decode(JobStatusResponse.self,
                           #"{"new_credits":2.5,"result_url":"https://cdn/x.mp4","status":"completed"}"#)
        XCTAssertEqual(s.newCredits, 2.5)
    }

    func testProcessingJobDecodes() throws {
        let s = try decode(JobStatusResponse.self,
                           #"{"message":"Waiting in queue...","progress":0,"status":"processing"}"#)
        XCTAssertEqual(s.status, "processing")
        XCTAssertEqual(s.progress, 0)
        XCTAssertNil(s.newCredits)
    }

    func testFailedJobDecodes() throws {
        let s = try decode(JobStatusResponse.self, #"{"status":"failed","error":"worker died"}"#)
        XCTAssertEqual(s.status, "failed")
        XCTAssertEqual(s.error, "worker died")
    }

    /// A junk value must not take the whole response down with it - status is
    /// the field that decides whether a render is finished.
    func testUnreadableCreditsDoesNotSinkTheResponse() throws {
        let s = try decode(JobStatusResponse.self,
                           #"{"new_credits":"not-a-number","status":"completed","result_url":"https://cdn/x.mp4"}"#)
        XCTAssertEqual(s.status, "completed")
        XCTAssertNil(s.newCredits)
        XCTAssertEqual(s.resultURL, "https://cdn/x.mp4")
    }

    /// The literal bytes /api/sam2/status returned for a real finished render
    /// on 2026-08-28. If this decodes, the app sees "completed" and collects
    /// the video instead of spinning until the 20 minute deadline.
    func testExactLivePayloadFromAFinishedRender() throws {
        let json = #"{"new_credits":1.0,"result_url":"https://markz.humblewoslayer.workers.dev/results/1787929801_81855dc7-f8ad-4dcc-98c9-5f141e1dfd88_sam2_removed.mp4","status":"completed"}"#
        let s = try decode(JobStatusResponse.self, json)
        XCTAssertEqual(s.status, "completed")
        XCTAssertEqual(s.newCredits, 1.0)
        XCTAssertNotNil(s.resultURL)
    }

    /// The same response in the shape that used to ship, so the app stays
    /// correct even if an old server is redeployed.
    func testExactLivePayloadInTheOldStringShape() throws {
        let json = #"{"new_credits":"1.00","result_url":"https://cdn/x.mp4","status":"completed"}"#
        XCTAssertEqual(try decode(JobStatusResponse.self, json).status, "completed")
    }

    // MARK: - Apple private relay

    /// Hide My Email gives an address whose local part is random. Printing it
    /// under someone's name looks like a bug in the app.
    func testPrivateRelayAddressIsNotShownRaw() {
        XCTAssertEqual(ProfileView.displayEmail("2kgpsd59vw@privaterelay.appleid.com"), "Hidden by Apple")
        XCTAssertEqual(ProfileView.displayEmail("K2JPSD59VW@PrivateRelay.AppleID.com"), "Hidden by Apple")
    }

    func testOrdinaryAddressIsShownAsIs() {
        XCTAssertEqual(ProfileView.displayEmail("hasan@markremoverai.com"), "hasan@markremoverai.com")
        XCTAssertEqual(ProfileView.displayEmail("chw2017@gmail.com"), "chw2017@gmail.com")
    }

    func testMissingAddressRendersEmpty() {
        XCTAssertEqual(ProfileView.displayEmail(nil), "")
        XCTAssertEqual(ProfileView.displayEmail(""), "")
    }
}
