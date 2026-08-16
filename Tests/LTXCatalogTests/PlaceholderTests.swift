// SPDX-License-Identifier: Apache-2.0
import Testing
@testable import LTXCatalog

@Suite("LTXCatalog")
struct LTXCatalogPlaceholderTests {
    @Test("target builds and links")
    func builds() { #expect(true) }
}
