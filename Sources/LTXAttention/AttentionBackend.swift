// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// The attention backend seam.
///
/// This is also where spatio-temporal guidance lives, and it is worth stating
/// the shape of it here because the obvious implementation is wrong: an
/// STG-perturbed block returns the value projection alone. `q_norm`, `k_norm`
/// and the entire RoPE application are skipped with it -- applying RoPE and
/// then discarding the result is not equivalent. See FRAGILE_CONTRACTS.md #6.
public enum AttentionBackend {
    public static let placeholder = true
}
