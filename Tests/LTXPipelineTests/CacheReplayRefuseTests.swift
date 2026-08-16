// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
@testable import LTXPipeline

@Suite("cache vs replay taps")
struct CacheReplayRefuseTests {

    @Test("dense cache and no observer is legal")
    func denseIsLegal() throws {
        try Pipeline.refuseCacheWithReplayTaps(hasReplayObserver: false, cacheCount: 0)
        try Pipeline.refuseCacheWithReplayTaps(hasReplayObserver: true, cacheCount: 0)
        try Pipeline.refuseCacheWithReplayTaps(hasReplayObserver: false, cacheCount: 4)
    }

    @Test("observer plus cache throws")
    func observerPlusCacheThrows() {
        #expect(throws: Pipeline.Failure.self) {
            try Pipeline.refuseCacheWithReplayTaps(hasReplayObserver: true, cacheCount: 1)
        }
        #expect(Pipeline.Failure.cacheWithReplayTaps.description.contains("omit taps"))
    }
}
