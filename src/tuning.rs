//! Runtime tuning for the High-mode postprocess passes (review finding A2).
//!
//! Every threshold that used to be a hard-coded `const` block inside an
//! `apply_*_postprocess` pass lives here as a field on [`PostprocessTuning`],
//! with `Default` reproducing the historical values. All luma/delta thresholds
//! are in 8-bit units; ratios are unitless; distances/gaps are in frames.
//!
//! Two auxiliary forward-similarity delta ceilings intentionally stay as
//! `const`s in `lib.rs` (`TEXT_CARD_MAX_SIMILARITY_DELTA_8BIT`,
//! `DARK_OCCLUSION_MAX_FORWARD_DELTA_8BIT`): the forward-similarity search may
//! substitute non-exact lower bounds for candidate deltas strictly above
//! `FORWARD_SIMILARITY_AUXILIARY_MAX_THRESHOLD_8BIT`, and that invariant is
//! enforced with a compile-time assertion which runtime options cannot honor.

/// Tuning for `apply_text_card_cluster_postprocess`: suppresses runs of weak
/// importance cuts produced by static text cards / title sequences.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TextCardClusterTuning {
    /// Minimum cuts in a run before the run is suppressed.
    pub min_cuts: usize,
    /// Maximum frame gap that keeps consecutive weak cuts in one run.
    pub max_gap: usize,
    /// Maximum cost ratio a cut may have and still count as "weak".
    pub max_cost_ratio: f64,
    /// Maximum average luma of a weak text-card cut.
    pub max_luma_8bit: f64,
    /// Maximum spatial importance ratio of a weak text-card cut.
    pub max_imp_ratio: f64,
    /// Minimum global importance ratio of a weak text-card cut.
    pub min_global_imp_ratio: f64,
    /// Maximum transient-similarity delta that marks the cut as a repeat.
    pub max_transient_delta_8bit: f64,
}

impl Default for TextCardClusterTuning {
    fn default() -> Self {
        Self {
            min_cuts: 3,
            max_gap: 130,
            max_cost_ratio: 0.40,
            max_luma_8bit: 35.0,
            max_imp_ratio: 4.0,
            min_global_imp_ratio: 3.8,
            max_transient_delta_8bit: 3.0,
        }
    }
}

/// Tuning for `apply_fast_motion_micro_split_postprocess`: removes pairs of
/// spurious splits inside fast-motion passages that end in a strong real cut.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FastMotionTuning {
    /// Maximum frames between the two suppressed micro-splits.
    pub pair_max_gap: usize,
    /// Maximum frames between the second split and the strong exit cut.
    pub pair_max_exit_gap: usize,
    /// Minimum frames between the preceding cut and the first split.
    pub pair_min_previous_gap: usize,
    /// Maximum cost ratio of a suppressible micro-split.
    pub max_cost_ratio: f64,
    /// Minimum global importance ratio of a suppressible micro-split.
    pub min_global_imp_ratio: f64,
    /// Minimum spatial importance ratio of a suppressible micro-split.
    pub min_imp_ratio: f64,
    /// Minimum average luma of a suppressible micro-split.
    pub min_luma_8bit: f64,
    /// Minimum transient-similarity delta of a suppressible micro-split.
    pub min_transient_delta_8bit: f64,
    /// Minimum cost ratio of the strong exit cut that anchors the pattern.
    pub exit_min_cost_ratio: f64,
}

impl Default for FastMotionTuning {
    fn default() -> Self {
        Self {
            pair_max_gap: 24,
            pair_max_exit_gap: 80,
            pair_min_previous_gap: 80,
            max_cost_ratio: 0.35,
            min_global_imp_ratio: 4.0,
            min_imp_ratio: 2.7,
            min_luma_8bit: 45.0,
            min_transient_delta_8bit: 6.0,
            exit_min_cost_ratio: 1.2,
        }
    }
}

/// Tuning for `apply_dark_occlusion_postprocess`: drops importance cuts caused
/// by objects briefly occluding dark scenes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DarkOcclusionTuning {
    /// Maximum cost ratio of a suppressible occlusion cut.
    pub max_cost_ratio: f64,
    /// Maximum average luma of a suppressible occlusion cut.
    pub max_luma_8bit: f64,
    /// Minimum spatial importance ratio of a suppressible occlusion cut.
    pub min_imp_ratio: f64,
    /// Minimum global importance ratio of a suppressible occlusion cut.
    pub min_global_imp_ratio: f64,
    /// Minimum static bad-block coverage of a suppressible occlusion cut.
    pub min_bad_block_ratio: f64,
    /// Maximum static good-block coverage of a suppressible occlusion cut.
    pub max_good_block_ratio: f64,
    /// Maximum transient-similarity delta marking the frames as a repeat.
    pub max_transient_delta_8bit: f64,
}

impl Default for DarkOcclusionTuning {
    fn default() -> Self {
        Self {
            max_cost_ratio: 0.25,
            max_luma_8bit: 25.0,
            min_imp_ratio: 3.5,
            min_global_imp_ratio: 4.5,
            min_bad_block_ratio: 0.70,
            max_good_block_ratio: 0.002,
            max_transient_delta_8bit: 4.0,
        }
    }
}

/// Shared geometry/density gates for the dark and sparse scene-peak recovery
/// passes (`apply_dark_scene_peak_recovery_postprocess` /
/// `apply_sparse_scene_peak_recovery_postprocess`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ScenePeakGeometryTuning {
    /// Minimum surrounding scene length that may receive a recovered cut.
    pub min_scene_len: usize,
    /// Radius over which the frame must be the local cost-ratio peak.
    pub local_radius: usize,
    /// Radius of the near cut-density window.
    pub density_radius: usize,
    /// Radius of the wide cut-density window.
    pub wide_density_radius: usize,
    /// Maximum cuts allowed inside the near density window.
    pub max_local_cuts: usize,
    /// Maximum cuts allowed inside the wide density window.
    pub max_wide_cuts: usize,
    /// Maximum cost ratio of the neighbouring cuts.
    pub max_adjacent_cost_ratio: f64,
}

impl Default for ScenePeakGeometryTuning {
    fn default() -> Self {
        Self {
            min_scene_len: 120,
            local_radius: 5,
            density_radius: 500,
            wide_density_radius: 2000,
            max_local_cuts: 10,
            max_wide_cuts: 35,
            max_adjacent_cost_ratio: 5.0,
        }
    }
}

/// Score gates for the dark scene-peak recovery pass.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DarkScenePeakTuning {
    /// Shared geometry/density gates.
    pub geometry: ScenePeakGeometryTuning,
    /// Minimum cost ratio of a recoverable dark peak.
    pub min_cost_ratio: f64,
    /// Maximum cost ratio of a recoverable dark peak.
    pub max_cost_ratio: f64,
    /// Maximum average luma of a recoverable dark peak.
    pub max_luma_8bit: f64,
    /// Minimum spatial importance ratio of a recoverable dark peak.
    pub min_imp_ratio: f64,
    /// Minimum global importance ratio of a recoverable dark peak.
    pub min_global_imp_ratio: f64,
    /// Minimum static bad-block coverage of a recoverable dark peak.
    pub min_bad_block_ratio: f64,
    /// Maximum static good-block coverage of a recoverable dark peak.
    pub max_good_block_ratio: f64,
}

impl Default for DarkScenePeakTuning {
    fn default() -> Self {
        Self {
            geometry: ScenePeakGeometryTuning::default(),
            min_cost_ratio: 0.25,
            max_cost_ratio: 0.45,
            max_luma_8bit: 50.0,
            min_imp_ratio: 1.3,
            min_global_imp_ratio: 1.4,
            min_bad_block_ratio: 0.62,
            max_good_block_ratio: 0.005,
        }
    }
}

/// Score gates for the sparse scene-peak recovery pass.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseScenePeakTuning {
    /// Shared geometry/density gates.
    pub geometry: ScenePeakGeometryTuning,
    /// Minimum cost ratio of a recoverable sparse peak.
    pub min_cost_ratio: f64,
    /// Maximum cost ratio of a recoverable sparse peak.
    pub max_cost_ratio: f64,
    /// Maximum average luma of a recoverable sparse peak.
    pub max_luma_8bit: f64,
    /// Minimum spatial importance ratio of a recoverable sparse peak.
    pub min_imp_ratio: f64,
    /// Minimum global importance ratio of a recoverable sparse peak.
    pub min_global_imp_ratio: f64,
    /// Minimum static bad-block coverage of a recoverable sparse peak.
    pub min_bad_block_ratio: f64,
    /// Maximum static good-block coverage of a recoverable sparse peak.
    pub max_good_block_ratio: f64,
}

impl Default for SparseScenePeakTuning {
    fn default() -> Self {
        Self {
            geometry: ScenePeakGeometryTuning::default(),
            min_cost_ratio: 0.25,
            max_cost_ratio: 1.0,
            max_luma_8bit: 90.0,
            min_imp_ratio: 1.5,
            min_global_imp_ratio: 1.8,
            min_bad_block_ratio: 0.60,
            max_good_block_ratio: 0.10,
        }
    }
}

/// Tuning for `apply_refined_sparse_peak_postprocess`: shifts/recovers sparse
/// peaks using frame-signature evidence and prunes short A-B-A echoes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RefinedSparsePeakTuning {
    /// Minimum distance enforced between refined cuts.
    pub min_distance: usize,
    /// Maximum forward shift applied to a sparse-peak boundary.
    pub shift_max_distance: usize,
    /// Minimum cost ratio of a refined peak.
    pub min_cost_ratio: f64,
    /// Maximum cost ratio of a refined peak.
    pub max_cost_ratio: f64,
    /// Minimum spatial importance ratio of a refined peak.
    pub min_imp_ratio: f64,
    /// Minimum global importance ratio of a refined peak.
    pub min_global_imp_ratio: f64,
    /// Minimum average luma of a refined peak.
    pub min_luma_8bit: f64,
    /// Maximum average luma of a refined peak.
    pub max_luma_8bit: f64,
    /// Minimum static bad-block coverage of a refined peak.
    pub min_bad_block_ratio: f64,
    /// Maximum static good-block coverage of a refined peak.
    pub max_good_block_ratio: f64,
    /// Minimum signature delta against the previous frame (a real edge).
    pub min_edge_delta_8bit: f64,
    /// Maximum signature delta that still counts as a repeated composition.
    pub max_repeat_delta_8bit: f64,
    /// Minimum signature delta that counts as a distinct composition.
    pub min_distinct_delta_8bit: f64,
    /// Radius of the near cut-density window.
    pub density_radius: usize,
    /// Radius of the wide cut-density window.
    pub wide_density_radius: usize,
    /// Maximum cuts allowed inside the near density window.
    pub max_local_cuts: usize,
    /// Maximum cuts allowed inside the wide density window.
    pub max_wide_cuts: usize,
    /// Maximum cost ratio of the neighbouring cuts.
    pub max_adjacent_cost_ratio: f64,
    /// Maximum A-B-A segment length eligible for echo suppression.
    pub aba_max_segment_len: usize,
}

impl Default for RefinedSparsePeakTuning {
    fn default() -> Self {
        Self {
            min_distance: 40,
            shift_max_distance: 40,
            min_cost_ratio: 0.65,
            max_cost_ratio: 1.05,
            min_imp_ratio: 3.0,
            min_global_imp_ratio: 3.5,
            min_luma_8bit: 55.0,
            max_luma_8bit: 90.0,
            min_bad_block_ratio: 0.50,
            max_good_block_ratio: 0.09,
            min_edge_delta_8bit: 10.0,
            max_repeat_delta_8bit: 5.0,
            min_distinct_delta_8bit: 8.0,
            density_radius: 500,
            wide_density_radius: 2000,
            max_local_cuts: 12,
            max_wide_cuts: 40,
            max_adjacent_cost_ratio: 5.0,
            aba_max_segment_len: 130,
        }
    }
}

/// Tuning for `apply_aba_return_recovery_postprocess`: re-inserts the missing
/// return boundary of an A-B-A pattern using frame signatures.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AbaReturnTuning {
    /// Minimum cost ratio of a recoverable return frame.
    pub min_cost_ratio: f64,
    /// Maximum cost ratio of a recoverable return frame.
    pub max_cost_ratio: f64,
    /// Minimum spatial importance ratio of a recoverable return frame.
    pub min_imp_ratio: f64,
    /// Minimum global importance ratio of a recoverable return frame.
    pub min_global_imp_ratio: f64,
    /// Maximum average luma of a recoverable return frame.
    pub max_luma_8bit: f64,
    /// Maximum static good-block coverage of a recoverable return frame.
    pub max_good_block_ratio: f64,
    /// Maximum signature delta between the return frame and the pre-B frame.
    pub max_signature_delta_8bit: f64,
    /// Minimum signature delta across each claimed boundary edge.
    pub min_edge_delta_8bit: f64,
    /// Minimum ratio of edge deltas to the return (repeat) delta.
    pub min_edge_ratio: f64,
}

impl Default for AbaReturnTuning {
    fn default() -> Self {
        Self {
            min_cost_ratio: 0.25,
            max_cost_ratio: 1.05,
            min_imp_ratio: 2.0,
            min_global_imp_ratio: 3.0,
            max_luma_8bit: 90.0,
            max_good_block_ratio: 0.10,
            max_signature_delta_8bit: 4.0,
            min_edge_delta_8bit: 5.0,
            min_edge_ratio: 2.0,
        }
    }
}

/// Tuning for `apply_aba_chain_compaction_postprocess`: merges long A-B-A-B
/// alternation chains back into two scenes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AbaChainTuning {
    /// Minimum alternating segments before a chain is compacted.
    pub min_segments: usize,
    /// Maximum signature delta for "same shot as two segments ago".
    pub max_signature_delta_8bit: f64,
    /// Minimum signature delta for "different shot than previous segment".
    pub min_different_delta_8bit: f64,
    /// Extra frames added on top of the forward-similarity window when
    /// deriving the maximum chain segment length.
    pub max_segment_extra: usize,
    /// Minimum cost ratio of a hidden (non-keyframe) chain boundary.
    pub min_hidden_cost_ratio: f64,
    /// Maximum cost ratio of a hidden chain boundary.
    pub max_hidden_cost_ratio: f64,
    /// Minimum spatial importance ratio of a hidden chain boundary.
    pub min_hidden_imp_ratio: f64,
    /// Minimum global importance ratio of a hidden chain boundary.
    pub min_hidden_global_imp_ratio: f64,
    /// Maximum average luma of a hidden chain boundary.
    pub max_hidden_luma_8bit: f64,
    /// Maximum static good-block coverage of a hidden chain boundary.
    pub max_hidden_good_block_ratio: f64,
    /// Minimum edge signature delta of a hidden chain boundary.
    pub min_hidden_edge_delta_8bit: f64,
}

impl Default for AbaChainTuning {
    fn default() -> Self {
        Self {
            min_segments: 4,
            max_signature_delta_8bit: 5.0,
            min_different_delta_8bit: 8.0,
            max_segment_extra: 50,
            min_hidden_cost_ratio: 0.25,
            max_hidden_cost_ratio: 1.2,
            min_hidden_imp_ratio: 2.0,
            min_hidden_global_imp_ratio: 3.0,
            max_hidden_luma_8bit: 95.0,
            max_hidden_good_block_ratio: 0.12,
            min_hidden_edge_delta_8bit: 5.0,
        }
    }
}

/// Tuning for `apply_forward_similarity_recovery_postprocess`: restores cuts
/// wrongly swallowed by forward-similarity suppression.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ForwardSimilarityRecoveryTuning {
    /// Minimum accepted-match delta before suppression looks doubtful.
    pub min_match_delta_8bit: f64,
    /// Minimum cost ratio of a recoverable suppressed cut.
    pub min_cost_ratio: f64,
    /// Minimum spatial importance ratio of a recoverable suppressed cut.
    pub min_imp_ratio: f64,
    /// Minimum global importance ratio of a recoverable suppressed cut.
    pub min_global_imp_ratio: f64,
    /// Minimum edge signature delta of a recoverable suppressed cut.
    pub min_edge_delta_8bit: f64,
    /// Maximum cuts restored per suppressed interval.
    pub max_frames: usize,
}

impl Default for ForwardSimilarityRecoveryTuning {
    fn default() -> Self {
        Self {
            min_match_delta_8bit: 8.0,
            min_cost_ratio: 1.2,
            min_imp_ratio: 5.0,
            min_global_imp_ratio: 6.0,
            min_edge_delta_8bit: 20.0,
            max_frames: 3,
        }
    }
}

/// Tuning for `apply_text_boundary_shift_postprocess`: moves boundaries that
/// min-distance pinned a few frames away from the true text-card change.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TextBoundaryShiftTuning {
    /// Minimum cost ratio of a forward-shift candidate.
    pub min_forward_cost_ratio: f64,
    /// Minimum spatial importance ratio of a forward-shift candidate.
    pub min_forward_imp_ratio: f64,
    /// Minimum global importance ratio of a forward-shift candidate.
    pub min_forward_global_imp_ratio: f64,
    /// Maximum average luma of a forward-shift candidate.
    pub max_forward_luma_8bit: f64,
    /// Minimum static bad-block coverage of a forward-shift candidate.
    pub min_forward_bad_block_ratio: f64,
    /// Maximum static good-block coverage of a forward-shift candidate.
    pub max_forward_good_block_ratio: f64,
    /// Minimum spatial importance ratio of a backward-shift candidate.
    pub min_backward_imp_ratio: f64,
    /// Minimum global importance ratio of a backward-shift candidate.
    pub min_backward_global_imp_ratio: f64,
    /// Maximum average luma of a backward-shift candidate.
    pub max_backward_luma_8bit: f64,
    /// Maximum cost ratio of the displaced next cut.
    pub max_next_cost_ratio: f64,
    /// Minimum average luma of the displaced next cut.
    pub min_next_luma_8bit: f64,
    /// Minimum global-importance margin over the displaced next cut.
    pub min_global_imp_margin: f64,
}

impl Default for TextBoundaryShiftTuning {
    fn default() -> Self {
        Self {
            min_forward_cost_ratio: 1.0,
            min_forward_imp_ratio: 8.0,
            min_forward_global_imp_ratio: 8.0,
            max_forward_luma_8bit: 40.0,
            min_forward_bad_block_ratio: 0.75,
            max_forward_good_block_ratio: 0.02,
            min_backward_imp_ratio: 4.5,
            min_backward_global_imp_ratio: 8.0,
            max_backward_luma_8bit: 35.0,
            max_next_cost_ratio: 1.0,
            min_next_luma_8bit: 45.0,
            min_global_imp_margin: 1.0,
        }
    }
}

/// Tuning for `apply_static_credits_postprocess`: keeps only the first cut of
/// long runs of mostly-static credit-roll cuts.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StaticCreditsTuning {
    /// Minimum cuts in a run before the run is compacted.
    pub min_run_cuts: usize,
    /// Maximum frame gap that keeps consecutive cuts in one run.
    pub max_gap: usize,
    /// Maximum average luma of a credits cut.
    pub max_luma_8bit: f64,
    /// Minimum spatial importance ratio of a credits cut.
    pub min_imp_ratio: f64,
    /// Minimum global importance ratio of a credits cut.
    pub min_global_imp_ratio: f64,
    /// Minimum static good-block coverage of a credits cut.
    pub min_good_block_ratio: f64,
    /// Maximum static bad-block coverage of a credits cut.
    pub max_bad_block_ratio: f64,
}

impl Default for StaticCreditsTuning {
    fn default() -> Self {
        Self {
            min_run_cuts: 5,
            max_gap: 210,
            max_luma_8bit: 45.0,
            min_imp_ratio: 6.0,
            min_global_imp_ratio: 8.0,
            min_good_block_ratio: 0.35,
            max_bad_block_ratio: 0.65,
        }
    }
}

/// Tuning for `apply_micro_scene_compaction_postprocess`: collapses scenes the
/// detector produced that are too short to be real scenes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MicroSceneTuning {
    /// Scenes shorter than this many frames lose their weaker boundary
    /// (`<= 1` disables the pass).
    pub min_scene_len: usize,
}

impl Default for MicroSceneTuning {
    fn default() -> Self {
        Self { min_scene_len: 4 }
    }
}

/// Tuning for `apply_local_peak_recovery_postprocess`: recovers cuts in
/// smooth low-texture scenes (fog, haze, gradients) where the absolute
/// importance metric stays far below the cut thresholds, but the cut frame is
/// a sharp outlier against its local neighborhood.
///
/// Tuned on the dragon fog reels (frames 73902..74792 and 133037..160978):
/// in-scene fog frames sit at an importance cost baseline with
/// `cost_ratio == 0` and zero backward cost, while the true cuts spike the
/// importance cost 3-13x above the local median with nonzero forward and
/// backward cost. Global flashes also spike, but shift the average luma by
/// 12+ (8-bit) versus at most ~4 across a fog cut, hence the luma-delta gate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LocalPeakRecoveryTuning {
    /// Minimum spatial importance ratio of a recoverable peak (noise floor).
    pub min_imp_ratio: f64,
    /// Minimum cost ratio of a recoverable peak.
    pub min_cost_ratio: f64,
    /// Minimum backward-adjusted cost as a fraction of the frame threshold.
    pub min_backward_cost_ratio: f64,
    /// Maximum static good-block coverage of a recoverable peak.
    pub max_good_block_ratio: f64,
    /// Maximum average-luma change against the previous frame (flash guard).
    pub max_luma_delta_8bit: f64,
    /// Minimum importance-cost multiple over both immediate neighbors.
    pub min_peak_over_neighbors: f64,
    /// Minimum importance-cost multiple over the local window median.
    pub min_peak_over_median: f64,
    /// Radius of the local median window (the peak and its immediate
    /// neighbors are excluded from the median).
    pub median_window_radius: usize,
    /// Minimum number of frames required in the median window.
    pub min_median_samples: usize,
    /// Minimum distance kept from existing cuts; also collapses candidate
    /// runs to their earliest frame (echoes trail the true peak).
    pub min_distance_to_cut: usize,
    /// Radius of the anti-strobe window. Rhythmic content (letter morphs,
    /// lightning shimmer) shows several distinct comparable peaks around the
    /// candidate; a fog cut is a lone outlier with at most one echo or
    /// motion swell nearby.
    pub strobe_radius: usize,
    /// Fraction of the candidate's importance cost a nearby frame must reach
    /// to count as a comparable peak.
    pub strobe_trigger_ratio: f64,
    /// Consecutive triggering frames closer than this merge into one event
    /// (a broad motion swell is one event, not many).
    pub strobe_merge_gap: usize,
    /// Maximum number of comparable peak events tolerated in the window.
    pub max_strobe_events: usize,
    /// Radius of the near cut-density window.
    pub density_radius: usize,
    /// Radius of the wide cut-density window.
    pub wide_density_radius: usize,
    /// Maximum cuts allowed inside the near density window.
    pub max_local_cuts: usize,
    /// Maximum cuts allowed inside the wide density window.
    pub max_wide_cuts: usize,
}

impl Default for LocalPeakRecoveryTuning {
    fn default() -> Self {
        Self {
            min_imp_ratio: 0.8,
            min_cost_ratio: 0.08,
            min_backward_cost_ratio: 0.15,
            max_good_block_ratio: 0.10,
            max_luma_delta_8bit: 8.0,
            min_peak_over_neighbors: 2.0,
            min_peak_over_median: 3.0,
            median_window_radius: 20,
            min_median_samples: 8,
            min_distance_to_cut: 25,
            strobe_radius: 24,
            strobe_trigger_ratio: 0.6,
            strobe_merge_gap: 3,
            max_strobe_events: 1,
            density_radius: 500,
            wide_density_radius: 2000,
            max_local_cuts: 20,
            max_wide_cuts: 70,
        }
    }
}

/// Tuning for `apply_signature_rescue_postprocess`: recovers cuts wrongly
/// vetoed by the importance-cut good-block gate.
///
/// In dark low-texture material (the loki1 supermarket reel, journey/f1 dark
/// interiors) flat blocks "match" across a real cut, inflating
/// `static_good_block_ratio` past the gate while the composition — the
/// mean-normalized spatial light distribution — clearly changes. Flicker and
/// flashes scale brightness without moving the composition, so requiring a
/// large normalized-signature delta re-admits the real cuts only.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SignatureRescueTuning {
    /// Minimum spatial importance ratio of a rescuable frame.
    pub min_imp_ratio: f64,
    /// Minimum cost ratio of a rescuable frame.
    pub min_cost_ratio: f64,
    /// Maximum cost ratio — above this the good gate is skipped anyway.
    pub max_cost_ratio: f64,
    /// Minimum static good-block ratio — below this the good gate never
    /// fired, so the frame is not a gate victim.
    pub min_good_block_ratio: f64,
    /// Maximum static good-block ratio of a rescuable frame.
    pub max_good_block_ratio: f64,
    /// Minimum mean absolute delta of the mean-normalized luma signatures
    /// against the previous frame (dimensionless; ~0.02 within a scene,
    /// 0.06-0.18 across flicker, 0.15+ across real cuts).
    pub min_norm_signature_delta: f64,
    /// Reject the rescue when at least this fraction of structure-bearing
    /// blocks still correlates across the transition: a partial flash or a
    /// brief occlusion moves the light composition while most of the frame's
    /// structure stays matched, whereas a real cut decorrelates it.
    pub max_structure_match_ratio: f64,
    /// The structure gate only applies when at least this fraction of active
    /// blocks carries structure; below it the match ratio is noise.
    pub min_structure_coverage: f64,
    /// Anti-strobe window radius (see `LocalPeakRecoveryTuning`).
    pub strobe_radius: usize,
    /// Anti-strobe trigger ratio.
    pub strobe_trigger_ratio: f64,
    /// Anti-strobe event merge gap.
    pub strobe_merge_gap: usize,
    /// Maximum tolerated comparable-peak events.
    pub max_strobe_events: usize,
    /// Minimum distance kept from existing cuts (earliest-wins collapse).
    pub min_distance_to_cut: usize,
    /// Radius of the near cut-density window.
    pub density_radius: usize,
    /// Radius of the wide cut-density window.
    pub wide_density_radius: usize,
    /// Maximum cuts allowed inside the near density window.
    pub max_local_cuts: usize,
    /// Maximum cuts allowed inside the wide density window.
    pub max_wide_cuts: usize,
}

impl Default for SignatureRescueTuning {
    fn default() -> Self {
        Self {
            min_imp_ratio: 2.9,
            min_cost_ratio: 0.2,
            max_cost_ratio: 0.55,
            min_good_block_ratio: 0.05,
            max_good_block_ratio: 0.20,
            min_norm_signature_delta: 0.15,
            max_structure_match_ratio: 0.7,
            min_structure_coverage: 0.01,
            strobe_radius: 24,
            strobe_trigger_ratio: 0.6,
            strobe_merge_gap: 3,
            max_strobe_events: 1,
            min_distance_to_cut: 25,
            density_radius: 500,
            wide_density_radius: 2000,
            max_local_cuts: 20,
            max_wide_cuts: 75,
        }
    }
}

/// Aggregated tuning for every High-mode postprocess pass.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PostprocessTuning {
    /// Text-card cut-run suppression.
    pub text_card: TextCardClusterTuning,
    /// Fast-motion micro-split suppression.
    pub fast_motion: FastMotionTuning,
    /// Dark-occlusion cut suppression.
    pub dark_occlusion: DarkOcclusionTuning,
    /// Dark scene-peak cut recovery.
    pub dark_scene_peak: DarkScenePeakTuning,
    /// Sparse scene-peak cut recovery.
    pub sparse_scene_peak: SparseScenePeakTuning,
    /// Signature-refined sparse-peak shifting/recovery.
    pub refined_sparse_peak: RefinedSparsePeakTuning,
    /// Local-outlier cut recovery for smooth (fog/haze) scenes.
    pub local_peak: LocalPeakRecoveryTuning,
    /// Good-gate victim recovery via normalized-signature composition change.
    pub signature_rescue: SignatureRescueTuning,
    /// A-B-A return-boundary recovery.
    pub aba_return: AbaReturnTuning,
    /// A-B-A-B alternation-chain compaction.
    pub aba_chain: AbaChainTuning,
    /// Recovery of cuts wrongly swallowed by forward similarity.
    pub forward_similarity_recovery: ForwardSimilarityRecoveryTuning,
    /// Min-distance boundary shifting for text cards.
    pub text_boundary_shift: TextBoundaryShiftTuning,
    /// Static credit-roll cut-run compaction.
    pub static_credits: StaticCreditsTuning,
    /// Degenerate micro-scene collapse.
    pub micro_scene: MicroSceneTuning,
}
