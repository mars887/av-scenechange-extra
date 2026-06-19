//! Scenechange detection tool based on rav1e's scene detection code.
//! It is focused around detecting scenechange points that will be optimal
//! for an encoder to place keyframes. It may not be the best tool
//! if your use case is to generate scene changes as a human would
//! interpret them--for that there are other tools such as `SCXvid` and `WWXD`.

mod analyze;
#[macro_use]
mod cpu;
mod data;
mod math;
mod options;

/// Hidden re-exports of internal items for benchmarking only.
/// Not part of the stable public API.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod _bench_internals {
    pub use crate::{
        analyze::{
            estimate_importance_block_difference,
            estimate_inter_costs,
            estimate_intra_costs,
        },
        data::FrameMEStats,
        math::Fixed,
    };
}

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{
        Arc,
        Condvar,
        Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, Sender, channel, sync_channel},
    },
    thread,
    time::{Duration, Instant},
};

pub use av_decoders::{self, Decoder};
pub use num_rational::Rational32;
use smallvec::SmallVec;
use v_frame::{frame::Frame, pixel::Pixel};

pub use crate::{
    analyze::{
        ForwardSimilarityCandidate,
        ForwardSimilarityCandidateDecision,
        SceneChangeDetector,
        ScenecutDecision,
        ScenecutResult,
    },
    options::{
        DetectionOptionOverride,
        ParseDetectionOptionOverrideError,
        detection_option_override_names,
    },
};

const FRAME_PREFETCH_DEPTH: usize = 8;
#[cfg(not(test))]
const PARALLEL_MIN_CHUNK_FRAMES: usize = 256;
#[cfg(test)]
const PARALLEL_MIN_CHUNK_FRAMES: usize = 24;
const PARALLEL_SYNC_MATCHES: usize = 2;
const PARALLEL_FORWARD_SUPPRESSED_SYNC_MATCHES: usize = 3;
const PARALLEL_READER_TARGET_BUFFER_BYTES: usize = 384 * 1024 * 1024;
/// Frame-count ceiling on a single worker's streamed read-ahead window. This is
/// only a backstop against a pathologically tiny per-frame payload demanding
/// millions of `BTreeMap` entries; the real bound is the byte budget above.
/// Raised well past the old 512 because, after the P2/P3 store reduction, a Fast
/// 1080p frame is only ~32 KiB, so 512 frames used barely ~16 MiB of the 384 MiB
/// budget — that artificial cap, not the byte budget, was what kept the reader
/// from decoding far enough ahead to feed later chunk workers (P1).
const PARALLEL_READER_MAX_BUFFER_FRAMES: usize = 16 * 1024;
/// Per-frame bookkeeping overhead (Arc control block, `BTreeMap` node, key)
/// folded into the byte-budget divisor so a very small pixel payload cannot
/// inflate the resident frame count without bound (P1).
const PARALLEL_READER_FRAME_ENTRY_OVERHEAD_BYTES: usize = 256;
const PARALLEL_READER_WAIT: Duration = Duration::from_millis(20);
const FRAME_REF_INLINE_CAPACITY: usize = 96;
/// Upper bound on the dense progress-dedup bitmap allocated in
/// [`reconcile_parallel_workers`]. That bitmap only deduplicates per-frame
/// progress callbacks — out-of-range indices are ignored and the authoritative
/// totals come from the keyframe set and `actual_frame_limit` — so capping it
/// cannot change detection results. It exists solely to stop a sentinel/huge
/// `frame_limit` (e.g. `usize::MAX`, a valid serial-API "no limit") from
/// requesting a multi-gigabyte allocation (C7). `1 << 24` ≈ 16.7M frames
/// (~77h @ 60fps) exceeds any real video, so legitimate inputs are unaffected.
const PARALLEL_PROGRESS_TRACK_CAP: usize = 1 << 24;
/// Version marker for diagnostics fields emitted by this fork.
pub const DIAGNOSTICS_VERSION: &str = "av-scenechange-extra-high-postprocess-diagnostics-v9";

/// Options for opt-in parallel scene detection.
#[derive(Debug, Clone, Copy)]
pub struct ParallelDetectionOptions {
    /// Number of analysis workers. `0` picks a conservative automatic value;
    /// `1` disables parallel detection.
    pub workers: usize,
}

impl Default for ParallelDetectionOptions {
    #[inline]
    fn default() -> Self {
        Self { workers: 0 }
    }
}

/// Options determining how to run scene change detection.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectionOptions {
    /// The speed of detection algorithm to use.
    /// Slower algorithms are more accurate/better for use in encoders.
    pub analysis_speed: SceneDetectionSpeed,
    /// Enabling this will utilize heuristics to avoid scenecuts
    /// that are too close to each other.
    /// This is generally useful if you want scenecut detection
    /// for use in an encoder.
    /// If you want a raw list of scene changes, you should disable this.
    pub detect_flashes: bool,
    /// The minimum distance between two scene changes.
    pub min_scenecut_distance: Option<usize>,
    /// The maximum distance between two scene changes.
    pub max_scenecut_distance: Option<usize>,
    /// The distance to look ahead in the video
    /// for scene flash detection and optional forward similarity checks.
    ///
    /// If forward similarity is enabled, the effective lookahead is at least
    /// the configured forward similarity window.
    pub lookahead_distance: usize,
    /// Optional tuning values for the detector internals.
    pub tuning: DetectionTuning,
    /// Disable nested cost-analysis rayon jobs when parallel scene detection
    /// uses at least this many workers. `0` keeps nested rayon enabled.
    pub disable_rayon_on_workers: usize,
}

impl Default for DetectionOptions {
    #[inline]
    fn default() -> Self {
        DetectionOptions {
            analysis_speed: SceneDetectionSpeed::Standard,
            detect_flashes: true,
            lookahead_distance: 5,
            min_scenecut_distance: None,
            max_scenecut_distance: None,
            tuning: DetectionTuning::default(),
            disable_rayon_on_workers: 0,
        }
    }
}

impl DetectionOptions {
    /// A higher quality preset intended for difficult content such as HDR,
    /// dark-to-dark cuts, and short transient scenes.
    #[inline]
    #[must_use]
    pub fn high_quality() -> Self {
        DetectionOptions {
            analysis_speed: SceneDetectionSpeed::High,
            lookahead_distance: 24,
            tuning: DetectionTuning::high_quality(),
            ..DetectionOptions::default()
        }
    }

    /// Returns the lookahead distance actually required by the selected
    /// flash-detection and forward-similarity settings.
    #[inline]
    #[must_use]
    pub fn effective_lookahead_distance(&self) -> usize {
        let flash_lookahead = if self.detect_flashes {
            self.lookahead_distance
        } else {
            1
        };
        if self.tuning.forward_similarity.enabled {
            flash_lookahead
                .max(
                    self.tuning.forward_similarity.frames.saturating_add(
                        self.tuning
                            .forward_similarity
                            .window_frames
                            .saturating_sub(1),
                    ),
                )
                .max(if self.tuning.transient_similarity.enabled {
                    self.tuning.transient_similarity.frames
                } else {
                    0
                })
        } else {
            flash_lookahead.max(if self.tuning.transient_similarity.enabled {
                self.tuning.transient_similarity.frames
            } else {
                0
            })
        }
    }
}

/// Internal scene detection tuning knobs.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectionTuning {
    /// How the fast detector threshold should scale with bit depth.
    pub fast_threshold_scale: FastThresholdScale,
    /// Fast detector threshold expressed in 8-bit luma units.
    pub fast_threshold_8bit: f64,
    /// How the standard/high importance threshold should be calculated.
    pub importance_mode: ImportanceThresholdMode,
    /// Base importance threshold expressed in 8-bit luma units.
    pub importance_threshold_8bit: f64,
    /// Lowest multiplier allowed for adaptive dark-scene importance
    /// thresholding.
    pub importance_min_factor: f64,
    /// Luma reference point, in 8-bit units, where adaptive importance reaches
    /// the full fixed threshold.
    pub importance_luma_ref_8bit: f64,
    /// How per-block importance deltas are aggregated.
    pub importance_aggregation: ImportanceAggregation,
    /// Allows strong cost-ratio peaks to bypass the importance gate.
    pub strong_cut_ratio: Option<f64>,
    /// Allows strong local importance-block peaks to trigger cuts in high mode.
    pub importance_cut_ratio: Option<f64>,
    /// Minimum cost-ratio evidence required for importance-driven cuts.
    pub importance_cut_min_cost_ratio: f64,
    /// Optional maximum luma, in 8-bit units, for importance-driven cuts.
    pub importance_cut_max_luma_8bit: Option<f64>,
    /// Optional maximum cost ratio for importance cuts above
    /// `importance_cut_dark_luma_high_8bit`.
    pub importance_cut_bright_max_cost_ratio: f64,
    /// Lower importance ratio accepted when a cut starts after quiet blocks.
    pub importance_cut_relaxed_ratio: Option<f64>,
    /// Maximum amount subtracted from relaxed importance ratio in dark scenes.
    pub importance_cut_dark_ratio_boost: f64,
    /// Lowest relaxed importance ratio after dark-scene adaptation.
    pub importance_cut_dark_min_ratio: f64,
    /// Luma where dark-scene relaxed threshold adaptation reaches full
    /// strength.
    pub importance_cut_dark_luma_low_8bit: f64,
    /// Luma where dark-scene relaxed threshold adaptation is disabled.
    pub importance_cut_dark_luma_high_8bit: f64,
    /// Lower cost-ratio accepted for abrupt dark-scene importance cuts.
    pub importance_cut_relaxed_min_cost_ratio: f64,
    /// Previous-frame importance ratio required for relaxed abrupt cuts.
    pub importance_cut_relaxed_max_previous_ratio: f64,
    /// Minimum ME residual coverage that can support relaxed cuts.
    pub importance_cut_min_me_bad_ratio: f64,
    /// Optional upper bound for well-matched ME blocks on importance cuts.
    ///
    /// High values usually mean localized motion, titles, or effects changed
    /// sharply while most of the frame still tracks well.
    pub importance_cut_max_me_good_ratio: f64,
    /// Optional A-B-A transient suppression.
    pub forward_similarity: ForwardSimilarityOptions,
    /// Optional two-sided masked similarity suppression for transient cuts.
    pub transient_similarity: TransientSimilarityOptions,
    /// Computes expensive motion-compensated cost diagnostics.
    ///
    /// The primary scene-cut signal remains zero-motion appearance cost. This
    /// switch is intended for analysis runs that need to inspect how much of a
    /// candidate can be explained by motion compensation.
    pub motion_cost_diagnostics: bool,
}

impl Default for DetectionTuning {
    #[inline]
    fn default() -> Self {
        Self {
            fast_threshold_scale: FastThresholdScale::Legacy,
            fast_threshold_8bit: 18.0,
            importance_mode: ImportanceThresholdMode::Fixed,
            importance_threshold_8bit: 7.0,
            importance_min_factor: 1.0,
            importance_luma_ref_8bit: 64.0,
            importance_aggregation: ImportanceAggregation::Mean,
            strong_cut_ratio: None,
            importance_cut_ratio: None,
            importance_cut_min_cost_ratio: 0.0,
            importance_cut_max_luma_8bit: None,
            importance_cut_bright_max_cost_ratio: 0.0,
            importance_cut_relaxed_ratio: None,
            importance_cut_dark_ratio_boost: 0.0,
            importance_cut_dark_min_ratio: 0.0,
            importance_cut_dark_luma_low_8bit: 25.0,
            importance_cut_dark_luma_high_8bit: 60.0,
            importance_cut_relaxed_min_cost_ratio: 0.0,
            importance_cut_relaxed_max_previous_ratio: 2.2,
            importance_cut_min_me_bad_ratio: 0.0,
            importance_cut_max_me_good_ratio: 0.0,
            forward_similarity: ForwardSimilarityOptions::default(),
            transient_similarity: TransientSimilarityOptions::default(),
            motion_cost_diagnostics: false,
        }
    }
}

impl DetectionTuning {
    /// Tuning preset used by [`DetectionOptions::high_quality`].
    #[inline]
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            fast_threshold_scale: FastThresholdScale::SampleRange,
            importance_mode: ImportanceThresholdMode::AdaptiveLuma,
            importance_min_factor: 0.35,
            importance_aggregation: ImportanceAggregation::SpatialTemporalTopBlocks {
                previous_percent: 0.10,
                current_percent: 0.15,
                next_percent: 0.10,
                region_cols: 8,
                region_rows: 4,
            },
            strong_cut_ratio: Some(2.5),
            importance_cut_ratio: Some(3.2),
            importance_cut_min_cost_ratio: 0.20,
            importance_cut_max_luma_8bit: Some(64.0),
            importance_cut_bright_max_cost_ratio: 0.9,
            importance_cut_relaxed_ratio: Some(3.0),
            importance_cut_dark_ratio_boost: 0.65,
            importance_cut_dark_min_ratio: 2.35,
            importance_cut_dark_luma_low_8bit: 25.0,
            importance_cut_dark_luma_high_8bit: 60.0,
            importance_cut_relaxed_min_cost_ratio: 0.08,
            importance_cut_relaxed_max_previous_ratio: 2.2,
            importance_cut_min_me_bad_ratio: 0.15,
            importance_cut_max_me_good_ratio: 0.05,
            forward_similarity: ForwardSimilarityOptions {
                enabled: true,
                frames: 80,
                window_frames: 3,
                min_offset: 4,
                threshold_8bit: 6.0,
                min_previous_scene_len: 24,
                min_cut_cost_ratio: 0.12,
                relaxed_threshold_8bit: 8.5,
                relaxed_min_cost_ratio: 1.0,
                relaxed_max_cost_ratio: 3.0,
                relaxed_importance_min_cost_ratio: 0.45,
                relaxed_min_imp_block_ratio: 3.6,
                relaxed_max_return_cost_ratio: 2.5,
                relaxed_max_return_cost_ratio_multiplier: 4.0,
                flash_return_without_candidate: true,
                flash_return_frames: 40,
                flash_return_min_cost_ratio: 1.0,
                mask_percent: 0.20,
                mask_region_cols: 8,
                mask_region_rows: 4,
                chroma_weight: 0.25,
                require_return_candidate: true,
                suppress_inside: true,
            },
            transient_similarity: TransientSimilarityOptions {
                enabled: true,
                frames: 10,
                threshold_8bit: 6.0,
                dark_threshold_8bit: 4.0,
                dark_luma_low_8bit: 25.0,
                dark_luma_high_8bit: 60.0,
                mask_percent: 0.20,
            },
            ..DetectionTuning::default()
        }
    }
}

/// Controls how the fast detector threshold scales for high bit depth inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum FastThresholdScale {
    /// Historical av-scenechange behavior: threshold * bit_depth / 8.
    Legacy,
    /// Scale by the actual sample range, e.g. 10-bit uses 1023 / 255.
    SampleRange,
}

/// Controls how the importance block threshold is calculated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ImportanceThresholdMode {
    /// Use the fixed historical threshold.
    Fixed,
    /// Lower the importance threshold for dark frames.
    AdaptiveLuma,
}

/// Controls how per-block importance deltas are reduced to a frame score.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ImportanceAggregation {
    /// Average all importance block deltas.
    Mean,
    /// Average current-frame block deltas selected from top blocks in the
    /// previous, current, and next adjacent-frame comparisons.
    TemporalTopBlocks {
        /// Fraction of top previous-comparison blocks to reuse.
        previous_percent: f64,
        /// Fraction of top current-comparison blocks to use.
        current_percent: f64,
        /// Fraction of top next-comparison blocks to reuse.
        next_percent: f64,
    },
    /// Select top temporal blocks with a cap per spatial region so localized
    /// flashes cannot dominate the score.
    SpatialTemporalTopBlocks {
        /// Fraction of top previous-comparison blocks to reuse.
        previous_percent: f64,
        /// Fraction of top current-comparison blocks to use.
        current_percent: f64,
        /// Fraction of top next-comparison blocks to reuse.
        next_percent: f64,
        /// Number of horizontal spatial regions.
        region_cols: usize,
        /// Number of vertical spatial regions.
        region_rows: usize,
    },
}

/// Options for suppressing short A-B-A transient cuts.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ForwardSimilarityOptions {
    /// Enable forward similarity suppression.
    pub enabled: bool,
    /// Number of future frames to inspect.
    pub frames: usize,
    /// Number of frames to compare on each side of the transient segment.
    /// A value of 1 preserves the legacy single-frame comparison.
    #[cfg_attr(
        feature = "serialize",
        serde(default = "default_forward_similarity_window_frames")
    )]
    pub window_frames: usize,
    /// Minimum future offset before a frame can be accepted as a return. This
    /// avoids suppressing hard cuts just because one of the next few frames is
    /// still visually similar to the previous scene.
    pub min_offset: usize,
    /// Maximum segment similarity delta, in 8-bit units, considered a return
    /// to the previous scene. Uses masked luma block comparison when
    /// `mask_percent` is non-zero and may include weighted chroma delta.
    pub threshold_8bit: f64,
    /// Minimum length of the scene before the cut before forward similarity
    /// can suppress the cut. This avoids treating very short pre-rolls as the
    /// stable A side of an A-B-A pattern.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub min_previous_scene_len: usize,
    /// Minimum cost-ratio evidence required before forward similarity can
    /// suppress a cut.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub min_cut_cost_ratio: f64,
    /// Optional relaxed segment similarity delta for high-confidence cuts.
    /// A value of 0 disables the relaxed threshold.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub relaxed_threshold_8bit: f64,
    /// Cost ratio required to use the relaxed threshold unconditionally.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub relaxed_min_cost_ratio: f64,
    /// Maximum cut cost ratio that may use the relaxed threshold. Very large
    /// ratios often come from flashes, explosions, or exposure shifts; keep
    /// those on the strict threshold unless the visual match is already clear.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub relaxed_max_cost_ratio: f64,
    /// Cost ratio required to use the relaxed threshold for strong
    /// importance-driven cuts.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub relaxed_importance_min_cost_ratio: f64,
    /// Importance ratio required to use the relaxed threshold for strong
    /// importance-driven cuts.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub relaxed_min_imp_block_ratio: f64,
    /// Maximum return-candidate cost ratio accepted for relaxed matches unless
    /// the return candidate is still proportionate to the starting cut.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub relaxed_max_return_cost_ratio: f64,
    /// Maximum return-candidate/start-cut cost ratio accepted for relaxed
    /// matches whose return candidate is already above
    /// `relaxed_max_return_cost_ratio`.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub relaxed_max_return_cost_ratio_multiplier: f64,
    /// Allows a strong hard cut to be suppressed as a short flash return even
    /// when the detector did not find a separate future return-cut candidate.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub flash_return_without_candidate: bool,
    /// Maximum forward offset for flash returns without a return-cut candidate.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub flash_return_frames: usize,
    /// Minimum starting cut cost ratio for flash returns without a return-cut
    /// candidate.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub flash_return_min_cost_ratio: f64,
    /// Fraction of most volatile blocks to mask when checking the return.
    pub mask_percent: f64,
    /// Number of horizontal regions used to cap masked volatile blocks.
    #[cfg_attr(
        feature = "serialize",
        serde(default = "default_forward_similarity_mask_region")
    )]
    pub mask_region_cols: usize,
    /// Number of vertical regions used to cap masked volatile blocks.
    #[cfg_attr(
        feature = "serialize",
        serde(default = "default_forward_similarity_mask_region")
    )]
    pub mask_region_rows: usize,
    /// Extra chroma delta weight added to the luma similarity score.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub chroma_weight: f64,
    /// Only accept a return frame if there is also a plausible future cut
    /// candidate before that return. This keeps A-B-A suppression
    /// segment-aware while allowing the best matching return frame to be a
    /// stable interior frame after the B->A boundary.
    pub require_return_candidate: bool,
    /// Suppress additional cuts until the detected return frame.
    pub suppress_inside: bool,
}

const fn default_forward_similarity_window_frames() -> usize {
    1
}

const fn default_forward_similarity_mask_region() -> usize {
    1
}

impl Default for ForwardSimilarityOptions {
    #[inline]
    fn default() -> Self {
        Self {
            enabled: false,
            frames: 0,
            window_frames: 1,
            min_offset: 2,
            threshold_8bit: 6.0,
            min_previous_scene_len: 0,
            min_cut_cost_ratio: 0.0,
            relaxed_threshold_8bit: 0.0,
            relaxed_min_cost_ratio: 0.0,
            relaxed_max_cost_ratio: 0.0,
            relaxed_importance_min_cost_ratio: 0.0,
            relaxed_min_imp_block_ratio: 0.0,
            relaxed_max_return_cost_ratio: 0.0,
            relaxed_max_return_cost_ratio_multiplier: 0.0,
            flash_return_without_candidate: false,
            flash_return_frames: 0,
            flash_return_min_cost_ratio: 0.0,
            mask_percent: 0.0,
            mask_region_cols: 1,
            mask_region_rows: 1,
            chroma_weight: 0.0,
            require_return_candidate: false,
            suppress_inside: false,
        }
    }
}

/// Options for suppressing transient flash/local-motion cuts by comparing
/// masked frames before and after the candidate cut.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TransientSimilarityOptions {
    /// Enable two-sided masked similarity suppression.
    pub enabled: bool,
    /// Number of frames to inspect on each side of the candidate.
    pub frames: usize,
    /// Maximum masked luma delta, in 8-bit units, considered the same scene.
    pub threshold_8bit: f64,
    /// Dark-scene threshold. Lower values avoid suppressing true dark cuts
    /// whose backgrounds are naturally similar after masking.
    pub dark_threshold_8bit: f64,
    /// Luma where dark-scene threshold adaptation reaches full strength.
    pub dark_luma_low_8bit: f64,
    /// Luma where dark-scene threshold adaptation is disabled.
    pub dark_luma_high_8bit: f64,
    /// Fraction of most volatile blocks to mask from the similarity score.
    pub mask_percent: f64,
}

impl Default for TransientSimilarityOptions {
    #[inline]
    fn default() -> Self {
        Self {
            enabled: false,
            frames: 0,
            threshold_8bit: 6.0,
            dark_threshold_8bit: 6.0,
            dark_luma_low_8bit: 25.0,
            dark_luma_high_8bit: 60.0,
            mask_percent: 0.20,
        }
    }
}

/// Results from a scene change detection pass.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectionResults {
    /// The 0-indexed frame numbers where scene changes were detected.
    pub scene_changes: Vec<usize>,
    /// A map of scores for each frame. Some frames may not have a score.
    pub scores: BTreeMap<usize, ScenecutResult>,
    /// The total number of frames read.
    pub frame_count: usize,
    /// Average speed (FPS)
    pub speed: f64,
}

/// # Errors
///
/// - If using a Vapoursynth script that contains an unsupported video format.
#[inline]
pub fn new_detector<T: Pixel>(
    dec: &mut Decoder,
    opts: DetectionOptions,
) -> anyhow::Result<SceneChangeDetector<T>> {
    let video_details = dec.get_video_details();

    Ok(new_detector_from_video_details::<T>(
        video_details,
        opts,
        true,
    ))
}

fn new_detector_from_video_details<T: Pixel>(
    video_details: &av_decoders::VideoDetails,
    opts: DetectionOptions,
    use_cost_parallelism: bool,
) -> SceneChangeDetector<T> {
    let mut detector = SceneChangeDetector::new(
        (video_details.width, video_details.height),
        video_details.bit_depth,
        video_details.frame_rate.recip(),
        video_details.chroma_sampling,
        opts.effective_lookahead_distance(),
        opts.analysis_speed,
        opts.tuning,
        opts.min_scenecut_distance.unwrap_or(0),
        opts.max_scenecut_distance.unwrap_or(u32::MAX as usize),
    );
    detector.set_cost_parallelism(use_cost_parallelism);
    detector
}

/// Runs through a y4m video clip,
/// detecting where scene changes occur.
/// This is adjustable based on the `opts` parameters.
///
/// This is the preferred, simplified interface
/// for analyzing a whole clip for scene changes.
///
/// # Arguments
///
/// - `progress_callback`: An optional callback that will fire after each frame
///   is analyzed. Arguments passed in will be, in order, the number of frames
///   analyzed, and the number of keyframes detected. This is generally useful
///   for displaying progress, etc.
///
/// # Errors
///
/// - If using a Vapoursynth script that contains an unsupported video format.
///
/// # Panics
///
/// - If the effective lookahead distance is 0.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(skip_all, fields(frame_limit))
)]
#[inline]
pub fn detect_scene_changes<T: Pixel>(
    dec: &mut Decoder,
    opts: DetectionOptions,
    frame_limit: Option<usize>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> anyhow::Result<DetectionResults> {
    let effective_lookahead = opts.effective_lookahead_distance();
    let transient_history = if opts.tuning.transient_similarity.enabled {
        opts.tuning.transient_similarity.frames
    } else {
        0
    };
    let forward_similarity_history = if opts.tuning.forward_similarity.enabled {
        opts.tuning
            .forward_similarity
            .window_frames
            .saturating_sub(1)
    } else {
        0
    };
    let frame_history = transient_history.max(forward_similarity_history);
    assert!(effective_lookahead >= 1);

    let detector = new_detector::<T>(dec, opts)?;
    let (frame_tx, frame_rx) = sync_channel(FRAME_PREFETCH_DEPTH);
    let (progress_tx, progress_rx) = if progress_callback.is_some() {
        let (tx, rx) = channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let detection_handle = {
        let progress_tx = progress_tx;
        thread::spawn(move || -> anyhow::Result<DetectionResults> {
            let mut detector = detector;
            let mut frame_queue = FrameWindow::new(0);
            let mut keyframes = BTreeSet::new();
            keyframes.insert(0);
            let mut scores = BTreeMap::new();

            let start_time = Instant::now();
            let mut frameno = 0usize;
            loop {
                let mut next_input_frameno = frame_queue.next_frame();
                let max_needed =
                    (frameno + effective_lookahead + 1).min(frame_limit.unwrap_or(usize::MAX));

                while next_input_frameno < max_needed {
                    match frame_rx.recv() {
                        Ok(frame) => {
                            frame_queue.push_next(next_input_frameno, frame);
                            next_input_frameno += 1;
                        }
                        Err(_) => break,
                    }
                }

                let frame_set_start = frameno.saturating_sub(1);
                let frame_set = frame_queue.refs_from(frame_set_start, effective_lookahead + 2);
                if frame_set.len() < 2 {
                    break;
                }
                if frameno == 0 {
                    keyframes.insert(frameno);
                } else {
                    let previous_frame_set = frame_queue.refs_range(
                        frameno.saturating_sub(frame_history.saturating_add(1)),
                        frameno,
                    );
                    let (cut, score) = detector.analyze_next_frame_with_history(
                        &frame_set,
                        &previous_frame_set,
                        frameno,
                        *keyframes
                            .iter()
                            .last()
                            .expect("at least 1 keyframe should exist"),
                    );
                    if let Some(score) = score {
                        scores.insert(frameno, score);
                    }
                    if cut {
                        keyframes.insert(frameno);
                    }
                }

                drop(frame_set);
                let remove_before = frameno.saturating_sub(frame_history.saturating_add(1));
                frame_queue.prune_before(remove_before);

                frameno += 1;
                if let Some(ref progress_tx) = progress_tx {
                    let _ = progress_tx.send((frameno, keyframes.len()));
                }
                if let Some(frame_limit) = frame_limit
                    && frameno == frame_limit
                {
                    break;
                }
            }

            apply_scenechange_postprocess(opts, &mut keyframes, &mut scores);

            Ok(DetectionResults {
                scene_changes: keyframes.into_iter().collect(),
                frame_count: frameno,
                speed: frameno as f64 / start_time.elapsed().as_secs_f64(),
                scores,
            })
        })
    };

    let mut produced = 0usize;
    while frame_limit.map_or_else(|| true, |limit| produced < limit) {
        #[cfg(feature = "tracing")]
        let span = tracing::span!(tracing::Level::INFO, "read_video_frame");
        #[cfg(feature = "tracing")]
        let enter = span.enter();
        match dec.read_video_frame() {
            Ok(frame) => {
                produced += 1;
                if frame_tx.send(Arc::new(frame)).is_err() {
                    break;
                }
            }
            Err(av_decoders::DecoderError::EndOfFile) => break,
            Err(e) => {
                return Err(e.into());
            }
        }
        #[cfg(feature = "tracing")]
        drop(enter);
        #[cfg(feature = "tracing")]
        drop(span);

        if let (Some(progress_rx), Some(progress_fn)) = (&progress_rx, progress_callback) {
            while let Ok((frames, keyframe_count)) = progress_rx.try_recv() {
                progress_fn(frames, keyframe_count);
            }
        }
    }

    drop(frame_tx);

    if let (Some(progress_rx), Some(progress_fn)) = (&progress_rx, progress_callback) {
        while let Ok((frames, keyframe_count)) = progress_rx.try_recv() {
            progress_fn(frames, keyframe_count);
        }
    }

    let results = detection_handle
        .join()
        .map_err(|_| anyhow::anyhow!("scene detection thread panicked"))??;

    if let (Some(progress_rx), Some(progress_fn)) = (&progress_rx, progress_callback) {
        while let Ok((frames, keyframe_count)) = progress_rx.try_recv() {
            progress_fn(frames, keyframe_count);
        }
    }

    Ok(results)
}

/// Runs scene detection using multiple analysis workers over a shared
/// sequential frame stream.
///
/// This keeps decoding deterministic and performs the speculative work inside
/// this crate. Chunk workers start at evenly spaced frame numbers and continue
/// past the next chunk boundary until their output can be reconciled with the
/// next worker. If reconciliation cannot be proven, the left worker remains
/// authoritative and the result falls back toward the sequential path.
///
/// # Errors
///
/// Returns decoder errors, worker panics, or reconciliation errors.
#[inline]
pub fn detect_scene_changes_parallel<T: Pixel + Send + Sync + 'static>(
    dec: &mut Decoder,
    opts: DetectionOptions,
    frame_limit: Option<usize>,
    parallel: ParallelDetectionOptions,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> anyhow::Result<DetectionResults> {
    let video_details = *dec.get_video_details();
    let Some(frame_count) = resolve_parallel_frame_count(frame_limit, video_details.total_frames)
    else {
        return detect_scene_changes::<T>(dec, opts, frame_limit, progress_callback);
    };
    let workers = resolve_parallel_workers(parallel.workers, frame_count);
    let chunk_starts = parallel_chunk_starts(frame_count, workers);
    if chunk_starts.len() <= 1 {
        return detect_scene_changes::<T>(dec, opts, Some(frame_count), progress_callback);
    }
    let use_worker_cost_parallelism = parallel_worker_cost_parallelism(opts, workers);

    let start_time = Instant::now();
    let store = Arc::new(SharedFrameStore::<T>::new());
    let (tx, rx) = channel();
    let stop_after = (0..chunk_starts.len())
        .map(|_| Arc::new(AtomicUsize::new(usize::MAX)))
        .collect::<Vec<_>>();
    let needed_from = chunk_starts
        .iter()
        .copied()
        .map(|start| Arc::new(AtomicUsize::new(parallel_initial_fetch_start(start, opts))))
        .collect::<Vec<_>>();
    let (progress_tx, progress_rx) = if progress_callback.is_some() {
        let (tx, rx) = channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let frame_reducer = parallel_frame_reducer::<T>(opts, &video_details);
    let per_worker_buffered_frames =
        parallel_reader_buffer_frames::<T>(opts, &video_details, &frame_reducer);
    let use_indexed_reader = decoder_supports_indexed_frames(dec);
    let (frame_request_tx, frame_request_rx) = if use_indexed_reader {
        let (tx, rx) = channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    thread::scope(|scope| -> anyhow::Result<DetectionResults> {
        let worker_handles = chunk_starts
            .iter()
            .copied()
            .enumerate()
            .map(|(worker, start_frame)| {
                let worker_store = Arc::clone(&store);
                let worker_tx = tx.clone();
                let worker_stop = Arc::clone(&stop_after[worker]);
                let worker_needed_from = Arc::clone(&needed_from[worker]);
                let worker_frame_request_tx = frame_request_tx.clone();
                scope.spawn(move || {
                    let panic_store = Arc::clone(&worker_store);
                    let worker_analysis_tx = worker_tx.clone();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_parallel_worker::<T>(
                            worker,
                            start_frame,
                            frame_count,
                            &video_details,
                            opts,
                            worker_store,
                            worker_analysis_tx,
                            worker_stop,
                            worker_needed_from,
                            worker_frame_request_tx,
                            use_worker_cost_parallelism,
                        )
                    }));
                    let message = match result {
                        Ok(Ok(frame_count)) => ParallelWorkerMessage::Done {
                            worker,
                            frame_count,
                            error: None,
                        },
                        Ok(Err(error)) => {
                            // Normal worker error: fail the shared store so the
                            // inline streamed reader's `produced_or_error()?`
                            // returns promptly instead of decoding the rest of the
                            // video, and any sibling blocked in `store.get` is
                            // released (C4 defect 2).
                            panic_store
                                .fail(format!("scene detection worker {worker} failed: {error}"));
                            ParallelWorkerMessage::Done {
                                worker,
                                frame_count: start_frame,
                                error: Some(error.to_string()),
                            }
                        }
                        Err(payload) => {
                            // Worker panicked: convert the payload into the same
                            // failure-signalling path (fail the store + report to
                            // reconcile) so the API returns `Err` rather than
                            // re-throwing the panic out of `thread::scope` (C4
                            // defect 1).
                            let detail = parallel_panic_message(&*payload);
                            panic_store.fail(format!(
                                "scene detection worker {worker} panicked: {detail}"
                            ));
                            ParallelWorkerMessage::Done {
                                worker,
                                frame_count: start_frame,
                                error: Some(format!("worker {worker} panicked: {detail}")),
                            }
                        }
                    };
                    let _ = worker_tx.send(message);
                })
            })
            .collect::<Vec<_>>();
        let reader_tx = tx.clone();
        drop(tx);

        let reconcile_stop_after = stop_after.clone();
        let reconcile_chunk_starts = chunk_starts.clone();
        let reconcile_progress_tx = progress_tx.clone();
        let reconcile_handle = scope.spawn(move || {
            reconcile_parallel_workers(
                rx,
                opts,
                &reconcile_chunk_starts,
                frame_count,
                &reconcile_stop_after,
                reconcile_progress_tx,
            )
        });

        let read_result = if let Some(frame_request_rx) = frame_request_rx {
            drop(frame_request_tx);
            read_parallel_indexed_frames::<T>(
                dec,
                frame_count,
                &store,
                &needed_from,
                frame_request_rx,
                &worker_handles,
                &reconcile_handle,
                &progress_rx,
                progress_callback,
                &frame_reducer,
            )
        } else {
            read_parallel_streamed_frames::<T>(
                dec,
                frame_count,
                &store,
                &needed_from,
                per_worker_buffered_frames,
                &progress_rx,
                progress_callback,
                &frame_reducer,
            )
        };
        if let Ok(produced) = &read_result {
            store.finish(*produced);
            let _ = reader_tx.send(ParallelWorkerMessage::ReaderDone {
                frame_count: *produced,
            });
        }
        drop(reader_tx);
        drain_parallel_progress(&progress_rx, progress_callback);

        let reconcile_result = reconcile_handle
            .join()
            .map_err(|_| anyhow::anyhow!("scene detection reconciliation thread panicked"))
            .and_then(|inner| inner);
        drain_parallel_progress(&progress_rx, progress_callback);

        // Stop every worker, then ALWAYS join them before returning. Joining a
        // scoped thread consumes its panic payload, which is what prevents
        // `thread::scope` from re-throwing a worker panic as an API-level panic
        // (C4 defect 1). On failure the worker already called `store.fail`, so the
        // inline reader has already stopped and these joins return promptly.
        for stop in &stop_after {
            stop.store(0, Ordering::Release);
        }
        let mut worker_panic: Option<anyhow::Error> = None;
        for handle in worker_handles {
            if handle.join().is_err() && worker_panic.is_none() {
                worker_panic = Some(anyhow::anyhow!("scene detection worker thread panicked"));
            }
        }
        drain_parallel_progress(&progress_rx, progress_callback);

        // Prefer the reconcile error (it carries the specific per-worker failure
        // message forwarded via `Done`); fall back to a worker join panic.
        let mut results = reconcile_result?;
        if let Some(worker_panic) = worker_panic {
            return Err(worker_panic);
        }

        let produced = read_result?;
        results.frame_count = produced.min(results.frame_count);
        results.speed = results.frame_count as f64 / start_time.elapsed().as_secs_f64();
        if let Some(progress_fn) = progress_callback {
            progress_fn(results.frame_count, results.scene_changes.len());
        }
        Ok(results)
    })
}

/// Runs parallel scene detection with one indexed decoder instance per worker.
///
/// This is intended for random-access backends such as VapourSynth where a
/// single shared decoder environment can serialize otherwise independent
/// chunk workers. `frame_start` is an absolute source-frame offset used when
/// analyzing a sub-range such as an Av1an zone; returned scene-change frame
/// numbers remain relative to that sub-range, matching
/// [`detect_scene_changes`].
///
/// # Errors
///
/// Returns decoder construction errors, frame-read errors, worker panics, or
/// reconciliation errors.
#[cfg(feature = "vapoursynth")]
#[inline]
pub fn detect_scene_changes_parallel_with_decoders<T, F>(
    dec: &mut Decoder,
    opts: DetectionOptions,
    frame_start: usize,
    frame_limit: Option<usize>,
    parallel: ParallelDetectionOptions,
    make_decoder: F,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> anyhow::Result<DetectionResults>
where
    T: Pixel + Send + Sync + 'static,
    F: Fn(usize) -> anyhow::Result<Decoder> + Sync,
{
    let video_details = *dec.get_video_details();
    let available = video_details
        .total_frames
        .map(|total_frames| total_frames.saturating_sub(frame_start));
    let Some(frame_count) = resolve_parallel_frame_count(frame_limit, available) else {
        // No finite decodable length is known, so the range cannot be split into
        // chunks. The serial reader streams sequentially from the decoder's
        // current position (frame 0) and cannot honor `frame_start`, so analyzing
        // a non-zero sub-range here would silently return the wrong frames (C6).
        // Reject that rather than corrupt the result; `frame_start == 0` is the
        // whole stream, which serial handles correctly.
        if frame_start != 0 {
            return Err(anyhow::anyhow!(
                "parallel scene detection with decoder instances requires a known frame \
                 count when frame_start > 0"
            ));
        }
        return detect_scene_changes::<T>(dec, opts, frame_limit, progress_callback);
    };
    if frame_count <= 1 {
        // A 0- or 1-frame range has nothing to reconcile: the lone worker breaks
        // on `frame_set.len() < 2` before emitting any decision, so reconcile
        // never reaches `complete` and would otherwise return `Err` (C8). The
        // result is the unconditional keyframe 0 with `frame_count == 0`,
        // independent of `frame_start`, which the serial path produces directly.
        // (Single chunks of >= 2 frames stay on the worker+reconcile path below,
        // which already honors `frame_start`.)
        return detect_scene_changes::<T>(dec, opts, Some(frame_count), progress_callback);
    }
    let workers = resolve_parallel_workers(parallel.workers, frame_count);
    let chunk_starts = parallel_chunk_starts(frame_count, workers);
    let use_worker_cost_parallelism = parallel_worker_cost_parallelism(opts, workers);

    let start_time = Instant::now();
    let (tx, rx) = channel();
    let stop_after = (0..chunk_starts.len())
        .map(|_| Arc::new(AtomicUsize::new(usize::MAX)))
        .collect::<Vec<_>>();
    let (progress_tx, progress_rx) = if progress_callback.is_some() {
        let (tx, rx) = channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    thread::scope(|scope| -> anyhow::Result<DetectionResults> {
        let worker_handles = chunk_starts
            .iter()
            .copied()
            .enumerate()
            .map(|(worker, start_frame)| {
                let worker_tx = tx.clone();
                let worker_stop = Arc::clone(&stop_after[worker]);
                let make_decoder = &make_decoder;
                scope.spawn(move || {
                    let worker_analysis_tx = worker_tx.clone();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_parallel_indexed_decoder_worker::<T, F>(
                            worker,
                            start_frame,
                            frame_start,
                            frame_count,
                            &video_details,
                            opts,
                            worker_analysis_tx,
                            worker_stop,
                            make_decoder,
                            use_worker_cost_parallelism,
                        )
                    }));
                    let message = match result {
                        Ok(Ok(frame_count)) => ParallelWorkerMessage::Done {
                            worker,
                            frame_count,
                            error: None,
                        },
                        Ok(Err(error)) => ParallelWorkerMessage::Done {
                            worker,
                            frame_count: start_frame,
                            error: Some(error.to_string()),
                        },
                        Err(payload) => {
                            // Worker panicked: report it to reconcile as a failure
                            // so the API returns `Err` instead of re-throwing the
                            // panic out of `thread::scope`. Siblings are stopped by
                            // the epilogue's `stop_after = 0` once reconcile
                            // returns (C4 defect 1).
                            let detail = parallel_panic_message(&*payload);
                            ParallelWorkerMessage::Done {
                                worker,
                                frame_count: start_frame,
                                error: Some(format!("worker {worker} panicked: {detail}")),
                            }
                        }
                    };
                    let _ = worker_tx.send(message);
                })
            })
            .collect::<Vec<_>>();
        drop(tx);

        let reconcile_stop_after = stop_after.clone();
        let reconcile_chunk_starts = chunk_starts.clone();
        let reconcile_progress_tx = progress_tx.clone();
        let reconcile_handle = scope.spawn(move || {
            reconcile_parallel_workers(
                rx,
                opts,
                &reconcile_chunk_starts,
                frame_count,
                &reconcile_stop_after,
                reconcile_progress_tx,
            )
        });

        while !reconcile_handle.is_finished() {
            drain_parallel_progress(&progress_rx, progress_callback);
            thread::sleep(PARALLEL_READER_WAIT);
        }
        let reconcile_result = reconcile_handle
            .join()
            .map_err(|_| anyhow::anyhow!("scene detection reconciliation thread panicked"))
            .and_then(|inner| inner);
        drain_parallel_progress(&progress_rx, progress_callback);

        // Stop every worker, then ALWAYS join them before returning so a scoped
        // worker panic is consumed here (converted to `Err`) instead of being
        // re-thrown as an API-level panic when this closure returns (C4 defect 1).
        for stop in &stop_after {
            stop.store(0, Ordering::Release);
        }
        let mut worker_panic: Option<anyhow::Error> = None;
        for handle in worker_handles {
            if handle.join().is_err() && worker_panic.is_none() {
                worker_panic = Some(anyhow::anyhow!("scene detection worker thread panicked"));
            }
        }
        drain_parallel_progress(&progress_rx, progress_callback);

        let mut results = reconcile_result?;
        if let Some(worker_panic) = worker_panic {
            return Err(worker_panic);
        }

        results.speed = results.frame_count as f64 / start_time.elapsed().as_secs_f64();
        if let Some(progress_fn) = progress_callback {
            progress_fn(results.frame_count, results.scene_changes.len());
        }
        Ok(results)
    })
}

/// Resolves the concrete frame count to split across parallel workers.
///
/// `frame_limit` may carry the serial API's "no limit" sentinel (`usize::MAX`);
/// splitting that overflows [`parallel_chunk_starts`] and oversizes the
/// reconcile progress bitmap (C7). Treat the sentinel as "unbounded" and, when a
/// real decodable length is known (`total_frames`, already offset by any
/// `frame_start` at the call site), clamp the limit to it. Returns `None` when
/// no finite length is known, signalling the caller to fall back to the serial
/// path rather than splitting an unbounded range.
fn resolve_parallel_frame_count(
    frame_limit: Option<usize>,
    total_frames: Option<usize>,
) -> Option<usize> {
    let frame_limit = frame_limit.filter(|&limit| limit != usize::MAX);
    match (frame_limit, total_frames) {
        (Some(limit), Some(total)) => Some(limit.min(total)),
        (Some(limit), None) => Some(limit),
        (None, total) => total,
    }
}

fn resolve_parallel_workers(requested: usize, frame_count: usize) -> usize {
    if requested == 1 || frame_count < PARALLEL_MIN_CHUNK_FRAMES * 2 {
        return 1;
    }
    let available = thread::available_parallelism().map_or(1, usize::from);
    let workers = if requested == 0 {
        available.min(8)
    } else {
        requested
    };
    workers
        .max(1)
        .min(available.max(1))
        .min((frame_count / PARALLEL_MIN_CHUNK_FRAMES).max(1))
}

fn parallel_worker_cost_parallelism(opts: DetectionOptions, workers: usize) -> bool {
    opts.disable_rayon_on_workers == 0 || workers < opts.disable_rayon_on_workers
}

fn decoder_supports_indexed_frames(dec: &mut Decoder) -> bool {
    #[cfg(feature = "vapoursynth")]
    {
        dec.get_vapoursynth_impl().is_some()
    }
    #[cfg(not(feature = "vapoursynth"))]
    {
        let _ = dec;
        false
    }
}

#[expect(clippy::too_many_arguments)]
fn read_parallel_streamed_frames<T: Pixel>(
    dec: &mut Decoder,
    frame_count: usize,
    store: &SharedFrameStore<T>,
    needed_from: &[Arc<AtomicUsize>],
    per_worker_buffered_frames: usize,
    progress_rx: &Option<Receiver<(usize, usize)>>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
    frame_reducer: &analyze::ParallelFrameReducer<T>,
) -> anyhow::Result<usize> {
    let mut produced = 0usize;
    loop {
        if produced == frame_count {
            break Ok(produced);
        }
        wait_for_parallel_reader_capacity(
            store,
            needed_from,
            per_worker_buffered_frames,
            progress_rx,
            progress_callback,
        )?;
        match dec.read_video_frame() {
            Ok(frame) => {
                store.push(produced, Arc::new(frame_reducer.apply(frame)));
                produced += 1;
                prune_parallel_frame_store(store, needed_from);
                drain_parallel_progress(progress_rx, progress_callback);
            }
            Err(av_decoders::DecoderError::EndOfFile) => break Ok(produced),
            Err(e) => {
                store.fail(e.to_string());
                break Err(e.into());
            }
        }
    }
}

#[cfg(feature = "vapoursynth")]
#[expect(clippy::too_many_arguments)]
fn read_parallel_indexed_frames<'scope, T: Pixel>(
    dec: &mut Decoder,
    mut frame_count: usize,
    store: &SharedFrameStore<T>,
    needed_from: &[Arc<AtomicUsize>],
    frame_request_rx: Receiver<usize>,
    worker_handles: &[thread::ScopedJoinHandle<'scope, ()>],
    reconcile_handle: &thread::ScopedJoinHandle<'scope, anyhow::Result<DetectionResults>>,
    progress_rx: &Option<Receiver<(usize, usize)>>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
    frame_reducer: &analyze::ParallelFrameReducer<T>,
) -> anyhow::Result<usize> {
    loop {
        drain_parallel_progress(progress_rx, progress_callback);
        prune_parallel_indexed_frame_store(store, needed_from);

        if reconcile_handle.is_finished()
            || worker_handles.iter().all(|handle| handle.is_finished())
        {
            break;
        }

        match frame_request_rx.recv_timeout(PARALLEL_READER_WAIT) {
            Ok(frame) => {
                if frame >= frame_count || store.contains(frame) {
                    continue;
                }
                match dec.get_video_frame(frame) {
                    Ok(data) => {
                        store.push(frame, Arc::new(frame_reducer.apply(data)));
                        prune_parallel_indexed_frame_store(store, needed_from);
                    }
                    Err(av_decoders::DecoderError::EndOfFile) => {
                        // `frame` is the first non-decodable index, so the true
                        // decodable length is `frame` (the assumed `frame_count`
                        // over-estimated it — e.g. y4m with unknown total, or
                        // metadata that lied). Clamp `frame_count` down so future
                        // requests for indices >= `frame` are skipped by the guard
                        // above and we never re-hit EOF, and mark the store
                        // finished at the true count so `get(f')` returns
                        // `Ok(None)` for every `f' >= frame`. Do NOT return:
                        // slower earlier-chunk workers may still be blocked on
                        // valid frames `f' < frame` that nobody has decoded yet,
                        // and the reader must stay alive to serve them. The loop
                        // still terminates via its existing handle/channel
                        // conditions once all workers finish.
                        frame_count = frame_count.min(frame);
                        store.finish(frame);
                    }
                    Err(e) => {
                        store.fail(e.to_string());
                        return Err(e.into());
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    store.fail("indexed scene detection reader stopped".to_string());
    Ok(frame_count)
}

#[cfg(not(feature = "vapoursynth"))]
#[expect(clippy::too_many_arguments)]
fn read_parallel_indexed_frames<'scope, T: Pixel>(
    dec: &mut Decoder,
    frame_count: usize,
    store: &SharedFrameStore<T>,
    needed_from: &[Arc<AtomicUsize>],
    frame_request_rx: Receiver<usize>,
    worker_handles: &[thread::ScopedJoinHandle<'scope, ()>],
    reconcile_handle: &thread::ScopedJoinHandle<'scope, anyhow::Result<DetectionResults>>,
    progress_rx: &Option<Receiver<(usize, usize)>>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
    frame_reducer: &analyze::ParallelFrameReducer<T>,
) -> anyhow::Result<usize> {
    let _ = (
        dec,
        store,
        needed_from,
        frame_request_rx,
        worker_handles,
        reconcile_handle,
        progress_rx,
        progress_callback,
        frame_reducer,
    );
    Ok(frame_count)
}

fn parallel_chunk_starts(frame_count: usize, workers: usize) -> Vec<usize> {
    if workers <= 1 || frame_count == 0 {
        return vec![0];
    }
    // Widen the product to `u128` so a sentinel/huge `frame_count` (e.g. a
    // near-`usize::MAX` limit on an unknown-length source) cannot overflow
    // `idx * frame_count` before the divide. Byte-identical to
    // `idx * frame_count / workers` for every non-overflowing input; reordering
    // to `frame_count / workers * idx` instead would shift the boundaries (C7).
    let mut starts = (0..workers)
        .map(|idx| (idx as u128 * frame_count as u128 / workers as u128) as usize)
        .collect::<Vec<_>>();
    starts.dedup();
    if starts.first().copied() != Some(0) {
        starts.insert(0, 0);
    }
    starts
}

fn parallel_frame_history(opts: DetectionOptions) -> usize {
    let transient_history = if opts.tuning.transient_similarity.enabled {
        opts.tuning.transient_similarity.frames
    } else {
        0
    };
    let forward_similarity_history = if opts.tuning.forward_similarity.enabled {
        opts.tuning
            .forward_similarity
            .window_frames
            .saturating_sub(1)
    } else {
        0
    };
    transient_history.max(forward_similarity_history)
}

fn parallel_initial_fetch_start(start_frame: usize, opts: DetectionOptions) -> usize {
    start_frame.saturating_sub(parallel_frame_history(opts).saturating_add(1))
}

/// Builds the [`analyze::ParallelFrameReducer`] for these options. Centralized
/// so the reader, the buffer-budget estimate, and the worker `frames_pre_downscaled`
/// flag are all derived identically and cannot drift (P2/P3).
fn parallel_frame_reducer<T: Pixel>(
    opts: DetectionOptions,
    video_details: &av_decoders::VideoDetails,
) -> analyze::ParallelFrameReducer<T> {
    analyze::ParallelFrameReducer::new(
        (video_details.width, video_details.height),
        opts.analysis_speed,
        opts.tuning.forward_similarity.enabled,
        opts.tuning.transient_similarity.enabled,
    )
}

/// Per-worker streamed read-ahead budget, in frames. The streamed reader scales
/// the actually-allowed resident range by the number of *active* workers (see
/// [`parallel_streamed_reader_buffer_frames`]), so this is the window budget for
/// a single worker: the byte target divided by the post-reduction per-frame cost
/// (plus a small per-entry overhead), floored by one worker's own
/// lookahead+history window and capped only as a sanity backstop. Returning the
/// per-worker (not global) budget is half of the P1 fix; the other half is no
/// longer truncating it at 512 frames.
fn parallel_reader_buffer_frames<T: Pixel>(
    opts: DetectionOptions,
    video_details: &av_decoders::VideoDetails,
    reducer: &analyze::ParallelFrameReducer<T>,
) -> usize {
    let minimum = opts
        .effective_lookahead_distance()
        .saturating_add(parallel_frame_history(opts))
        .saturating_add(8);
    let frame_bytes = estimated_frame_bytes(video_details, reducer)
        .saturating_add(PARALLEL_READER_FRAME_ENTRY_OVERHEAD_BYTES)
        .max(1);
    let byte_limited = PARALLEL_READER_TARGET_BUFFER_BYTES / frame_bytes;
    byte_limited
        .max(minimum)
        .min(PARALLEL_READER_MAX_BUFFER_FRAMES.max(minimum))
}

/// Estimates the bytes a single frame occupies in the store *after* the P2/P3
/// reduction, so a smaller payload lets the byte budget buffer more frames
/// (which also relieves the P1 reader-serialization bottleneck).
fn estimated_frame_bytes<T: Pixel>(
    video_details: &av_decoders::VideoDetails,
    reducer: &analyze::ParallelFrameReducer<T>,
) -> usize {
    let bytes_per_sample = video_details.bit_depth.div_ceil(8).max(1);
    // Pre-downscaled store: luma only, shrunk by factor^2 (chroma also dropped).
    if let Some(factor) = reducer.scale_factor() {
        let luma_pixels =
            (video_details.width / factor).saturating_mul(video_details.height / factor);
        return luma_pixels.saturating_mul(bytes_per_sample).max(1);
    }
    let luma_pixels = video_details.width.saturating_mul(video_details.height);
    let chroma_pixels = if reducer.drops_chroma() {
        0
    } else {
        match video_details.chroma_sampling {
            v_frame::chroma::ChromaSubsampling::Yuv420 => luma_pixels / 2,
            v_frame::chroma::ChromaSubsampling::Yuv422 => luma_pixels,
            v_frame::chroma::ChromaSubsampling::Yuv444 => luma_pixels.saturating_mul(2),
            v_frame::chroma::ChromaSubsampling::Monochrome => 0,
        }
    };
    luma_pixels
        .saturating_add(chroma_pixels)
        .saturating_mul(bytes_per_sample)
}

fn wait_for_parallel_reader_capacity<T: Pixel>(
    store: &SharedFrameStore<T>,
    needed_from: &[Arc<AtomicUsize>],
    per_worker_buffered_frames: usize,
    progress_rx: &Option<Receiver<(usize, usize)>>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> anyhow::Result<()> {
    loop {
        drain_parallel_progress(progress_rx, progress_callback);
        // P1: size the resident range by how many workers are *currently* active,
        // not by a single global window. The old cap was one window measured from
        // `min(needed_from)` — which worker 0 pins near frame 0 — so the
        // forward-only reader never decoded far enough ahead to feed workers whose
        // chunks begin thousands of frames later; they starved in `store.get` and
        // the run collapsed to serial. Scaling the cap by the active-worker count
        // lets the reader reach the later chunks, while the byte budget keeps
        // total memory bounded (≈ `active_workers · target`, with `active_workers`
        // ≤ the chunk/worker count). This does NOT change which frames any worker
        // analyzes, only how far ahead the reader buffers, so results are
        // unchanged. Retention stays a contiguous prefix `[keep_from, produced]`,
        // so the forward-only "evicted frame is gone forever" hazard is avoided.
        let Some((keep_from, active_workers)) = parallel_active_needed_window(needed_from) else {
            return Ok(());
        };
        store.prune_before(keep_from);
        let produced = store.produced_or_error()?;
        let max_buffered_frames =
            parallel_streamed_reader_buffer_frames(per_worker_buffered_frames, active_workers);
        if produced.saturating_sub(keep_from) <= max_buffered_frames {
            return Ok(());
        }
        store.wait_for_change(PARALLEL_READER_WAIT);
    }
}

/// Total streamed resident-range budget: one [`parallel_reader_buffer_frames`]
/// window per active worker. The forward-only store cannot hold holes, so this is
/// the count of frames the reader may keep resident between the trailing active
/// worker (`keep_from`) and `produced`. `saturating_mul` so a sentinel per-worker
/// value cannot overflow. Peak pixel memory is therefore bounded by roughly
/// `active_workers · PARALLEL_READER_TARGET_BUFFER_BYTES`, and `active_workers` is
/// itself bounded by the worker/chunk count (≤ available cores). It does NOT hold
/// the whole video: a worker whose chunk lies beyond this range still waits until
/// the trailing worker retires and the window slides forward — i.e. very long
/// full-res inputs with chunk gaps larger than the budget remain partially
/// serialized (that needs indexed/per-worker decoders), but the common case where
/// several chunks fit the budget now overlaps instead of serializing (P1).
fn parallel_streamed_reader_buffer_frames(
    per_worker_buffered_frames: usize,
    active_workers: usize,
) -> usize {
    per_worker_buffered_frames.saturating_mul(active_workers.max(1))
}

fn prune_parallel_frame_store<T: Pixel>(
    store: &SharedFrameStore<T>,
    needed_from: &[Arc<AtomicUsize>],
) {
    let Some(keep_from) = minimum_parallel_needed_frame(needed_from) else {
        return;
    };
    store.prune_before(keep_from);
}

#[cfg(feature = "vapoursynth")]
fn prune_parallel_indexed_frame_store<T: Pixel>(
    store: &SharedFrameStore<T>,
    needed_from: &[Arc<AtomicUsize>],
) {
    let needed_frames = needed_from
        .iter()
        .map(|needed| needed.load(Ordering::Acquire))
        .filter(|&frame| frame != usize::MAX)
        .collect::<Vec<_>>();
    store.retain_only(&needed_frames);
}

fn minimum_parallel_needed_frame(needed_from: &[Arc<AtomicUsize>]) -> Option<usize> {
    parallel_active_needed_window(needed_from).map(|(keep_from, _)| keep_from)
}

/// Returns `(min over active needed_from, number of active workers)`, where an
/// "active" worker is one whose `needed_from` is not the `usize::MAX` retired
/// sentinel. `keep_from` is the streamed store's prune floor; the active count
/// drives the per-active-worker read-ahead budget (P1). Single pass over the
/// atomics so the throttle reads each slot once per wait iteration.
fn parallel_active_needed_window(needed_from: &[Arc<AtomicUsize>]) -> Option<(usize, usize)> {
    let mut keep_from = usize::MAX;
    let mut active_workers = 0usize;
    for needed in needed_from {
        let frame = needed.load(Ordering::Acquire);
        if frame == usize::MAX {
            continue;
        }
        keep_from = keep_from.min(frame);
        active_workers += 1;
    }
    (active_workers != 0).then_some((keep_from, active_workers))
}

fn drain_parallel_progress(
    progress_rx: &Option<Receiver<(usize, usize)>>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) {
    if let (Some(progress_rx), Some(progress_fn)) = (progress_rx, progress_callback) {
        while let Ok((frames, keyframe_count)) = progress_rx.try_recv() {
            progress_fn(frames, keyframe_count);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ParallelFrameDecision {
    is_scene_change: bool,
    score: Option<ScenecutResult>,
}

enum ParallelWorkerMessage {
    Frame {
        worker: usize,
        frame: usize,
        decision: ParallelFrameDecision,
    },
    Done {
        worker: usize,
        frame_count: usize,
        error: Option<String>,
    },
    ReaderDone {
        frame_count: usize,
    },
}

type FrameRefVec<'a, T> = SmallVec<[&'a Arc<Frame<T>>; FRAME_REF_INLINE_CAPACITY]>;

struct FrameWindow<T: Pixel> {
    start: usize,
    frames: VecDeque<Arc<Frame<T>>>,
}

impl<T: Pixel> FrameWindow<T> {
    fn new(start: usize) -> Self {
        Self {
            start,
            frames: VecDeque::new(),
        }
    }

    fn next_frame(&self) -> usize {
        self.start + self.frames.len()
    }

    fn push_next(&mut self, frame: usize, data: Arc<Frame<T>>) {
        debug_assert_eq!(frame, self.next_frame());
        if self.frames.is_empty() && frame != self.start {
            self.start = frame;
        }
        self.frames.push_back(data);
    }

    fn refs_from(&self, start: usize, take: usize) -> FrameRefVec<'_, T> {
        let mut refs = FrameRefVec::new();
        if take == 0 || self.frames.is_empty() {
            return refs;
        }
        let start = start.max(self.start);
        let offset = start - self.start;
        if offset >= self.frames.len() {
            return refs;
        }
        refs.extend(self.frames.iter().skip(offset).take(take));
        refs
    }

    fn refs_range(&self, start: usize, end: usize) -> FrameRefVec<'_, T> {
        if end <= start {
            return FrameRefVec::new();
        }
        let start = start.max(self.start);
        if end <= start {
            return FrameRefVec::new();
        }
        self.refs_from(start, end - start)
    }

    fn prune_before(&mut self, frame: usize) {
        if frame <= self.start {
            return;
        }
        let remove = (frame - self.start).min(self.frames.len());
        self.frames.drain(..remove);
        self.start += remove;
    }
}

struct SharedFrameStore<T: Pixel> {
    inner: Mutex<SharedFrameStoreInner<T>>,
    ready: Condvar,
}

struct SharedFrameStoreInner<T: Pixel> {
    frames: BTreeMap<usize, Arc<Frame<T>>>,
    produced: usize,
    finished: bool,
    error: Option<String>,
}

impl<T: Pixel> SharedFrameStore<T> {
    fn new() -> Self {
        Self {
            inner: Mutex::new(SharedFrameStoreInner {
                frames: BTreeMap::new(),
                produced: 0,
                finished: false,
                error: None,
            }),
            ready: Condvar::new(),
        }
    }

    fn push(&self, frame: usize, data: Arc<Frame<T>>) {
        let mut inner = self.inner.lock().expect("frame store lock poisoned");
        inner.frames.insert(frame, data);
        inner.produced = inner.produced.max(frame + 1);
        self.ready.notify_all();
    }

    fn prune_before(&self, frame: usize) {
        if frame == 0 {
            return;
        }
        let mut inner = self.inner.lock().expect("frame store lock poisoned");
        if inner
            .frames
            .keys()
            .next()
            .is_some_and(|&oldest| oldest < frame)
        {
            inner.frames = inner.frames.split_off(&frame);
        }
    }

    #[cfg(feature = "vapoursynth")]
    fn retain_only(&self, frames: &[usize]) {
        let mut inner = self.inner.lock().expect("frame store lock poisoned");
        inner.frames.retain(|frame, _| frames.contains(frame));
    }

    /// Non-blocking check whether the store currently holds `frame`.
    ///
    /// Used by the indexed reader to decide whether a requested frame must be
    /// (re-)decoded. Querying the store directly — rather than a separate
    /// permanently-growing `loaded` ledger — lets evicted frames be re-decoded
    /// when an earlier worker reaches a chunk overlap, which is what closes the
    /// C2 deadlock.
    #[cfg(feature = "vapoursynth")]
    fn contains(&self, frame: usize) -> bool {
        let inner = self.inner.lock().expect("frame store lock poisoned");
        inner.frames.contains_key(&frame)
    }

    fn produced_or_error(&self) -> anyhow::Result<usize> {
        let inner = self.inner.lock().expect("frame store lock poisoned");
        if let Some(error) = &inner.error {
            return Err(anyhow::anyhow!(
                "scene detection frame reader failed: {error}"
            ));
        }
        Ok(inner.produced)
    }

    fn wait_for_change(&self, timeout: Duration) {
        let inner = self.inner.lock().expect("frame store lock poisoned");
        let (guard, _) = self
            .ready
            .wait_timeout(inner, timeout)
            .expect("frame store lock poisoned");
        drop(guard);
    }

    fn notify_waiters(&self) {
        self.ready.notify_all();
    }

    fn finish(&self, produced: usize) {
        let mut inner = self.inner.lock().expect("frame store lock poisoned");
        inner.produced = inner.produced.max(produced);
        inner.finished = true;
        self.ready.notify_all();
    }

    fn fail(&self, error: String) {
        let mut inner = self.inner.lock().expect("frame store lock poisoned");
        inner.error = Some(error);
        inner.finished = true;
        self.ready.notify_all();
    }

    fn get(&self, frame: usize, frame_limit: usize) -> anyhow::Result<Option<Arc<Frame<T>>>> {
        if frame >= frame_limit {
            return Ok(None);
        }

        let mut inner = self.inner.lock().expect("frame store lock poisoned");
        loop {
            if let Some(data) = inner.frames.get(&frame) {
                return Ok(Some(Arc::clone(data)));
            }
            if let Some(error) = &inner.error {
                return Err(anyhow::anyhow!(
                    "scene detection frame reader failed: {error}"
                ));
            }
            if inner.finished && inner.produced <= frame {
                return Ok(None);
            }
            inner = self.ready.wait(inner).expect("frame store lock poisoned");
        }
    }
}

struct ParallelWorkerNeedGuard<T: Pixel> {
    needed_from: Arc<AtomicUsize>,
    store: Arc<SharedFrameStore<T>>,
}

impl<T: Pixel> ParallelWorkerNeedGuard<T> {
    fn new(needed_from: Arc<AtomicUsize>, store: Arc<SharedFrameStore<T>>) -> Self {
        Self { needed_from, store }
    }
}

impl<T: Pixel> Drop for ParallelWorkerNeedGuard<T> {
    fn drop(&mut self) {
        self.needed_from.store(usize::MAX, Ordering::Release);
        self.store.notify_waiters();
    }
}

#[expect(clippy::too_many_arguments)]
fn run_parallel_worker<T: Pixel>(
    worker: usize,
    start_frame: usize,
    frame_limit: usize,
    video_details: &av_decoders::VideoDetails,
    opts: DetectionOptions,
    store: Arc<SharedFrameStore<T>>,
    tx: Sender<ParallelWorkerMessage>,
    stop_after: Arc<AtomicUsize>,
    needed_from: Arc<AtomicUsize>,
    frame_request_tx: Option<Sender<usize>>,
    use_cost_parallelism: bool,
) -> anyhow::Result<usize> {
    let effective_lookahead = opts.effective_lookahead_distance();
    let frame_history = parallel_frame_history(opts);
    assert!(effective_lookahead >= 1);

    let initial_fetch_start = parallel_initial_fetch_start(start_frame, opts);
    let mut detector =
        new_detector_from_video_details::<T>(video_details, opts, use_cost_parallelism);
    detector.set_frames_pre_downscaled(
        parallel_frame_reducer::<T>(opts, video_details).is_prescaled(),
    );
    let mut frame_queue = FrameWindow::new(initial_fetch_start);
    let mut keyframes = BTreeSet::new();
    keyframes.insert(start_frame);

    let _needed_guard = ParallelWorkerNeedGuard::new(Arc::clone(&needed_from), Arc::clone(&store));
    needed_from.store(initial_fetch_start, Ordering::Release);
    store.notify_waiters();
    let mut frameno = start_frame;
    loop {
        let stop = stop_after.load(Ordering::Acquire);
        if frameno >= frame_limit || frameno >= stop {
            break;
        }
        let mut next_input_frameno = frame_queue.next_frame();
        let max_needed = (frameno + effective_lookahead + 1).min(frame_limit);

        while next_input_frameno < max_needed {
            needed_from.store(next_input_frameno, Ordering::Release);
            store.notify_waiters();
            if let Some(frame_request_tx) = &frame_request_tx {
                let _ = frame_request_tx.send(next_input_frameno);
            }
            match store.get(next_input_frameno, frame_limit)? {
                Some(frame) => {
                    frame_queue.push_next(next_input_frameno, frame);
                    next_input_frameno += 1;
                    needed_from.store(next_input_frameno, Ordering::Release);
                    store.notify_waiters();
                }
                None => break,
            }
        }

        let frame_set_start = frameno.saturating_sub(1);
        let frame_set = frame_queue.refs_from(frame_set_start, effective_lookahead + 2);
        if frame_set.len() < 2 {
            break;
        }

        let mut decision = ParallelFrameDecision {
            is_scene_change: false,
            score: None,
        };
        if worker == 0 && frameno == 0 {
            decision.is_scene_change = true;
        } else {
            let previous_frame_set = frame_queue.refs_range(
                frameno.saturating_sub(frame_history.saturating_add(1)),
                frameno,
            );
            let (cut, score) = detector.analyze_next_frame_with_history(
                &frame_set,
                &previous_frame_set,
                frameno,
                *keyframes
                    .iter()
                    .last()
                    .expect("at least 1 keyframe should exist"),
            );
            decision.score = score;
            if cut {
                keyframes.insert(frameno);
                decision.is_scene_change = true;
            }
        }

        if tx
            .send(ParallelWorkerMessage::Frame {
                worker,
                frame: frameno,
                decision,
            })
            .is_err()
        {
            break;
        }

        drop(frame_set);
        let remove_before = frameno.saturating_sub(frame_history.saturating_add(1));
        frame_queue.prune_before(remove_before);

        frameno += 1;
    }

    Ok(frameno)
}

#[cfg(feature = "vapoursynth")]
#[expect(clippy::too_many_arguments)]
fn run_parallel_indexed_decoder_worker<T, F>(
    worker: usize,
    start_frame: usize,
    frame_start: usize,
    mut frame_limit: usize,
    video_details: &av_decoders::VideoDetails,
    opts: DetectionOptions,
    tx: Sender<ParallelWorkerMessage>,
    stop_after: Arc<AtomicUsize>,
    make_decoder: &F,
    use_cost_parallelism: bool,
) -> anyhow::Result<usize>
where
    T: Pixel,
    F: Fn(usize) -> anyhow::Result<Decoder> + Sync,
{
    let mut source = make_decoder(worker)?;
    let frame_reducer = parallel_frame_reducer::<T>(opts, video_details);
    let effective_lookahead = opts.effective_lookahead_distance();
    let frame_history = parallel_frame_history(opts);
    assert!(effective_lookahead >= 1);

    let initial_fetch_start = parallel_initial_fetch_start(start_frame, opts);
    let mut detector =
        new_detector_from_video_details::<T>(video_details, opts, use_cost_parallelism);
    detector.set_frames_pre_downscaled(frame_reducer.is_prescaled());
    let mut frame_queue = FrameWindow::new(initial_fetch_start);
    let mut keyframes = BTreeSet::new();
    keyframes.insert(start_frame);

    let mut frameno = start_frame;
    loop {
        let stop = stop_after.load(Ordering::Acquire);
        if frameno >= frame_limit || frameno >= stop {
            break;
        }
        let mut next_input_frameno = frame_queue.next_frame();
        let max_needed = (frameno + effective_lookahead + 1).min(frame_limit);

        while next_input_frameno < max_needed {
            let source_frame = frame_start + next_input_frameno;
            let frame = match source.get_video_frame(source_frame) {
                Ok(frame) => frame,
                Err(av_decoders::DecoderError::EndOfFile) => {
                    // `next_input_frameno` is the first non-decodable
                    // (worker-relative) index, so the true decodable length is
                    // exactly `next_input_frameno` — `frame_limit` over-estimated
                    // it (e.g. y4m with unknown total, or metadata that lied).
                    // Unlike `detect_scene_changes_parallel`, this path has no
                    // reader thread, so reconcile never receives a `ReaderDone`
                    // otherwise: report the discovered real end ourselves so
                    // reconcile can clamp `actual_frame_limit` and actually
                    // complete (a graceful break alone would leave it stuck on the
                    // over-estimate). Then clamp our own `frame_limit` so the outer
                    // loop terminates promptly and this fetch never re-requests a
                    // past-EOF index. Analyze the frames already obtained; this is
                    // a graceful stop, NOT an error (C5).
                    let _ = tx.send(ParallelWorkerMessage::ReaderDone {
                        frame_count: next_input_frameno,
                    });
                    frame_limit = frame_limit.min(next_input_frameno);
                    break;
                }
                Err(error) => {
                    return Err(anyhow::anyhow!(
                        "worker {worker} failed to read frame {source_frame}: {error}"
                    ));
                }
            };
            frame_queue.push_next(next_input_frameno, Arc::new(frame_reducer.apply(frame)));
            next_input_frameno += 1;
        }

        let frame_set_start = frameno.saturating_sub(1);
        let frame_set = frame_queue.refs_from(frame_set_start, effective_lookahead + 2);
        if frame_set.len() < 2 {
            break;
        }

        let mut decision = ParallelFrameDecision {
            is_scene_change: false,
            score: None,
        };
        if worker == 0 && frameno == 0 {
            decision.is_scene_change = true;
        } else {
            let previous_frame_set = frame_queue.refs_range(
                frameno.saturating_sub(frame_history.saturating_add(1)),
                frameno,
            );
            let (cut, score) = detector.analyze_next_frame_with_history(
                &frame_set,
                &previous_frame_set,
                frameno,
                *keyframes
                    .iter()
                    .last()
                    .expect("at least 1 keyframe should exist"),
            );
            decision.score = score;
            if cut {
                keyframes.insert(frameno);
                decision.is_scene_change = true;
            }
        }

        if tx
            .send(ParallelWorkerMessage::Frame {
                worker,
                frame: frameno,
                decision,
            })
            .is_err()
        {
            break;
        }

        drop(frame_set);
        let remove_before = frameno.saturating_sub(frame_history.saturating_add(1));
        frame_queue.prune_before(remove_before);

        frameno += 1;
    }

    Ok(frameno)
}

#[derive(Default)]
struct ParallelWorkerOutput {
    frames: BTreeMap<usize, ParallelFrameDecision>,
    done: bool,
    frame_count: usize,
}

impl ParallelWorkerOutput {
    fn insert(&mut self, frame: usize, decision: ParallelFrameDecision) {
        self.frames.insert(frame, decision);
        self.frame_count = self.frame_count.max(frame + 1);
    }

    fn contiguous_end_from(&self, start: usize, frame_limit: usize) -> usize {
        let mut frame = start;
        while frame < frame_limit && self.frames.contains_key(&frame) {
            frame += 1;
        }
        frame
    }

    fn reaches_end_from(&self, start: usize, frame_limit: usize) -> bool {
        self.done && self.contiguous_end_from(start, frame_limit) >= frame_limit
    }

    fn prune_before(&mut self, frame: usize) {
        if frame == 0 {
            return;
        }
        if self
            .frames
            .keys()
            .next()
            .is_some_and(|&oldest| oldest < frame)
        {
            self.frames = self.frames.split_off(&frame);
        }
    }
}

/// Extracts a human-readable message from a panic payload captured by
/// `std::panic::catch_unwind`, handling the common `&str` / `String` cases.
fn parallel_panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn reconcile_parallel_workers(
    rx: Receiver<ParallelWorkerMessage>,
    opts: DetectionOptions,
    chunk_starts: &[usize],
    frame_limit: usize,
    stop_after: &[Arc<AtomicUsize>],
    progress_tx: Option<Sender<(usize, usize)>>,
) -> anyhow::Result<DetectionResults> {
    let mut outputs = (0..chunk_starts.len())
        .map(|_| ParallelWorkerOutput::default())
        .collect::<Vec<_>>();
    let mut canonical_worker = 0usize;
    let mut canonical_start = 0usize;
    let mut keyframes = BTreeSet::new();
    let mut scores = BTreeMap::new();
    let mut reported_progress = 0usize;
    // Progress-dedup bitmap only: out-of-range indices are ignored below and the
    // authoritative totals come from `keyframes` / `actual_frame_limit`, never
    // from this vector. Cap its length so a sentinel/huge `frame_limit` cannot
    // demand a multi-gigabyte (or aborting) allocation; real videos are far
    // smaller than the cap, so every frame is still tracked exactly (C7).
    let mut analyzed_frames = vec![false; frame_limit.min(PARALLEL_PROGRESS_TRACK_CAP)];
    let mut analyzed_count = 0usize;
    let mut actual_frame_limit = frame_limit;
    let mut complete = false;

    while let Ok(message) = rx.recv() {
        match message {
            ParallelWorkerMessage::Frame {
                worker,
                frame,
                decision,
            } => {
                if worker >= canonical_worker
                    && let Some(output) = outputs.get_mut(worker)
                {
                    output.insert(frame, decision);
                }
                if frame < analyzed_frames.len() && !analyzed_frames[frame] {
                    analyzed_frames[frame] = true;
                    analyzed_count += 1;
                    report_parallel_progress(
                        progress_tx.as_ref(),
                        &mut reported_progress,
                        analyzed_count.min(actual_frame_limit),
                        keyframes.len(),
                    );
                }
            }
            ParallelWorkerMessage::Done {
                worker,
                frame_count,
                error,
            } => {
                if let Some(error) = error {
                    return Err(anyhow::anyhow!(
                        "scene detection worker {worker} failed: {error}"
                    ));
                }
                if let Some(output) = outputs.get_mut(worker) {
                    output.done = true;
                    output.frame_count = output.frame_count.max(frame_count);
                }
            }
            ParallelWorkerMessage::ReaderDone { frame_count } => {
                // MIN-accumulate so multiple reporters converge to the true end.
                // The single-reader path (`detect_scene_changes_parallel`) sends
                // exactly one `ReaderDone`, and `actual_frame_limit` starts at
                // `frame_limit`, so this is equivalent to the previous
                // `frame_count.min(frame_limit)`. In `_with_decoders` several
                // workers may each report the (worker-relative) index at which
                // they hit EOF; the smallest is the real decodable length.
                actual_frame_limit = actual_frame_limit.min(frame_count);
            }
        }

        loop {
            let append_limit = chunk_starts
                .get(canonical_worker + 1)
                .copied()
                .unwrap_or(actual_frame_limit)
                .min(actual_frame_limit);
            let appended_until = append_parallel_authoritative_prefix(
                &mut outputs[canonical_worker],
                canonical_start,
                append_limit,
                &mut keyframes,
                &mut scores,
            )?;
            if appended_until > canonical_start {
                canonical_start = appended_until;
                report_parallel_progress(
                    progress_tx.as_ref(),
                    &mut reported_progress,
                    canonical_start,
                    keyframes.len(),
                );
            }

            if canonical_start >= actual_frame_limit {
                complete = true;
                break;
            }

            if canonical_worker + 1 >= outputs.len() {
                break;
            }

            let boundary = chunk_starts[canonical_worker + 1].min(actual_frame_limit);
            if canonical_start < boundary {
                break;
            }

            if let Some(handoff) = find_parallel_handoff(
                &outputs[canonical_worker],
                &outputs[canonical_worker + 1],
                boundary,
                opts,
                actual_frame_limit,
            ) {
                let previous_worker = canonical_worker;
                append_parallel_range(
                    &outputs[previous_worker],
                    canonical_start,
                    handoff,
                    &mut keyframes,
                    &mut scores,
                )?;
                stop_after[previous_worker].store(handoff, Ordering::Release);
                canonical_worker += 1;
                canonical_start = handoff;
                outputs[previous_worker].frames.clear();
                outputs[canonical_worker].prune_before(canonical_start);
                report_parallel_progress(
                    progress_tx.as_ref(),
                    &mut reported_progress,
                    canonical_start,
                    keyframes.len(),
                );
            } else {
                break;
            }
        }
        if !complete
            && outputs[canonical_worker].reaches_end_from(canonical_start, actual_frame_limit)
        {
            append_parallel_range(
                &outputs[canonical_worker],
                canonical_start,
                actual_frame_limit,
                &mut keyframes,
                &mut scores,
            )?;
            report_parallel_progress(
                progress_tx.as_ref(),
                &mut reported_progress,
                actual_frame_limit,
                keyframes.len(),
            );
            complete = true;
        }
        if complete {
            break;
        }
    }

    if !complete {
        return Err(anyhow::anyhow!(
            "parallel scene detection ended before the authoritative worker reached EOF"
        ));
    }

    apply_scenechange_postprocess(opts, &mut keyframes, &mut scores);

    Ok(DetectionResults {
        scene_changes: keyframes.into_iter().collect(),
        scores,
        frame_count: actual_frame_limit,
        speed: 0.0,
    })
}

fn append_parallel_authoritative_prefix(
    output: &mut ParallelWorkerOutput,
    start: usize,
    append_limit: usize,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) -> anyhow::Result<usize> {
    let end = output.contiguous_end_from(start, append_limit);
    if end > start {
        append_parallel_range(output, start, end, keyframes, scores)?;
        output.prune_before(end);
    }
    Ok(end)
}

fn append_parallel_range(
    output: &ParallelWorkerOutput,
    start: usize,
    end: usize,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) -> anyhow::Result<()> {
    for frame in start..end {
        let decision = output.frames.get(&frame).ok_or_else(|| {
            anyhow::anyhow!("parallel scene detection missing frame {frame} during reconciliation")
        })?;
        if decision.is_scene_change {
            keyframes.insert(frame);
        }
        if let Some(score) = decision.score {
            scores.insert(frame, score);
        }
    }
    Ok(())
}

fn report_parallel_progress(
    progress_tx: Option<&Sender<(usize, usize)>>,
    reported_progress: &mut usize,
    frames: usize,
    keyframes: usize,
) {
    if frames > *reported_progress {
        *reported_progress = frames;
        if let Some(progress_tx) = progress_tx {
            let _ = progress_tx.send((frames, keyframes));
        }
    }
}

fn find_parallel_handoff(
    left: &ParallelWorkerOutput,
    right: &ParallelWorkerOutput,
    boundary: usize,
    opts: DetectionOptions,
    frame_limit: usize,
) -> Option<usize> {
    let max_common = left
        .contiguous_end_from(boundary, frame_limit)
        .min(right.contiguous_end_from(boundary, frame_limit));
    if max_common <= boundary {
        return None;
    }

    let mut matched_cuts = 0usize;
    let mut stable_run = 0usize;
    let mut quiet_run = 0usize;
    let mut saw_forward_suppression = false;
    let quiet_required = no_cut_parallel_warmup(opts);

    for frame in boundary..max_common {
        let left_decision = left.frames.get(&frame)?;
        let right_decision = right.frames.get(&frame)?;
        let forward_suppressed = is_forward_similarity_suppressed(left_decision.score)
            || is_forward_similarity_suppressed(right_decision.score);
        let exact_match = left_decision == right_decision;
        saw_forward_suppression |= forward_suppressed;

        if exact_match {
            stable_run += 1;
        } else {
            stable_run = 0;
            matched_cuts = 0;
            quiet_run = 0;
            continue;
        }

        let required_matches = if saw_forward_suppression {
            PARALLEL_FORWARD_SUPPRESSED_SYNC_MATCHES
        } else {
            PARALLEL_SYNC_MATCHES
        };
        if left_decision.is_scene_change {
            matched_cuts += 1;
            quiet_run = 0;
        } else if forward_suppressed {
            quiet_run = 0;
        } else {
            quiet_run += 1;
        }

        if matched_cuts >= required_matches && stable_run >= quiet_required {
            return Some(frame + 1);
        }
        if opts.max_scenecut_distance.is_none() && quiet_run >= quiet_required {
            return Some(frame + 1);
        }
    }

    None
}

fn no_cut_parallel_warmup(opts: DetectionOptions) -> usize {
    let previous_scene_len = if opts.tuning.forward_similarity.enabled {
        opts.tuning.forward_similarity.min_previous_scene_len
    } else {
        0
    };
    opts.effective_lookahead_distance()
        .max(opts.min_scenecut_distance.unwrap_or(0))
        .max(previous_scene_len)
        .saturating_add(parallel_frame_history(opts))
        .saturating_add(8)
}

fn is_forward_similarity_suppressed(score: Option<ScenecutResult>) -> bool {
    score.is_some_and(|score| score.decision == ScenecutDecision::SuppressedForwardSimilarity)
}

#[derive(Debug, Clone, Copy)]
struct PostprocessForwardSimilarityMatch {
    return_frame: usize,
    return_candidate_frame: usize,
    delta: f64,
}

fn apply_scenechange_postprocess(
    opts: DetectionOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    apply_forward_similarity_postprocess(opts.tuning.forward_similarity, keyframes, scores);
    if opts.analysis_speed == SceneDetectionSpeed::High {
        apply_text_card_cluster_postprocess(keyframes, scores);
        apply_fast_motion_micro_split_postprocess(keyframes, scores);
        apply_dark_occlusion_postprocess(keyframes, scores);
        apply_dark_scene_peak_recovery_postprocess(opts, keyframes, scores);
        apply_sparse_scene_peak_recovery_postprocess(opts, keyframes, scores);
        apply_refined_sparse_peak_postprocess(opts, keyframes, scores);
        apply_aba_return_recovery_postprocess(opts, keyframes, scores);
        apply_aba_chain_compaction_postprocess(opts, keyframes, scores);
        apply_forward_similarity_recovery_postprocess(opts, keyframes, scores);
        apply_text_boundary_shift_postprocess(opts, keyframes, scores);
        apply_static_credits_postprocess(keyframes, scores);
    }
}

fn apply_forward_similarity_postprocess(
    options: ForwardSimilarityOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    if !options.enabled || !options.require_return_candidate || options.frames == 0 {
        return;
    }

    let candidates = keyframes
        .iter()
        .copied()
        .filter(|&frame| frame != 0)
        .chain(scores.iter().filter_map(|(&frame, score)| {
            (frame != 0 && score.decision == ScenecutDecision::SuppressedForwardSimilarity)
                .then_some(frame)
        }))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    for frame in candidates {
        let frame_is_keyframe = keyframes.contains(&frame);
        if !frame_is_keyframe
            && scores.get(&frame).map(|score| score.decision)
                != Some(ScenecutDecision::SuppressedForwardSimilarity)
        {
            continue;
        }

        let Some(score) = scores.get(&frame).copied() else {
            continue;
        };
        if frame_is_keyframe
            && !matches!(
                score.decision,
                ScenecutDecision::Cut | ScenecutDecision::CutImportance
            )
        {
            continue;
        }
        let previous_frame = keyframes.range(..frame).next_back().copied().unwrap_or(0);
        if frame_is_keyframe
            && !analyze::forward_similarity_start_allowed(options, score, frame - previous_frame)
        {
            continue;
        }

        let Some(similarity_match) =
            forward_similarity_postprocess_match(frame, options, score, scores)
        else {
            continue;
        };

        if frame_is_keyframe {
            keyframes.remove(&frame);
        }
        mark_forward_similarity_suppressed(
            scores,
            frame,
            similarity_match.return_frame,
            similarity_match.return_candidate_frame,
            Some(similarity_match.delta),
        );

        if options.suppress_inside {
            let inside = keyframes
                .range((frame + 1)..=similarity_match.return_frame)
                .copied()
                .collect::<Vec<_>>();
            for inside_frame in inside {
                keyframes.remove(&inside_frame);
                mark_forward_similarity_suppressed(
                    scores,
                    inside_frame,
                    similarity_match.return_frame,
                    similarity_match.return_candidate_frame,
                    None,
                );
            }
        }
    }
}

fn forward_similarity_postprocess_match(
    frame: usize,
    options: ForwardSimilarityOptions,
    score: ScenecutResult,
    scores: &BTreeMap<usize, ScenecutResult>,
) -> Option<PostprocessForwardSimilarityMatch> {
    let min_offset = options.min_offset.max(2);
    let max_offset = options
        .frames
        .saturating_add(options.window_frames.saturating_sub(1))
        .saturating_add(1);
    let threshold = analyze::forward_similarity_threshold_8bit(options, score);
    score
        .forward_similarity_candidates
        .iter()
        .flatten()
        .filter(|candidate| {
            candidate.delta <= threshold
                && candidate.offset >= min_offset
                && candidate.offset <= max_offset
                && candidate.frame > frame
        })
        .filter_map(|candidate| {
            let search_start = frame + min_offset - 1;
            if search_start > candidate.frame {
                return None;
            }
            let return_candidate_frame = scores
                .range(search_start..=candidate.frame)
                .rev()
                .find_map(|(&candidate_frame, candidate_score)| {
                    forward_similarity_return_score_passed(*candidate_score)
                        .then_some(candidate_frame)
                })?;
            let post_start_frame = candidate.frame + 1 - options.window_frames.max(1);
            if return_candidate_frame > post_start_frame {
                return None;
            }
            let return_candidate_score = scores.get(&return_candidate_frame).copied()?;
            if !analyze::forward_similarity_return_candidate_allowed(
                options,
                score,
                return_candidate_score,
                candidate.delta,
            ) {
                return None;
            }

            Some(PostprocessForwardSimilarityMatch {
                return_frame: candidate.frame,
                return_candidate_frame,
                delta: candidate.delta,
            })
        })
        .min_by(|left, right| {
            left.delta
                .partial_cmp(&right.delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn forward_similarity_return_score_passed(score: ScenecutResult) -> bool {
    matches!(
        score.decision,
        ScenecutDecision::Cut
            | ScenecutDecision::CutImportance
            | ScenecutDecision::SuppressedForwardSimilarity
            | ScenecutDecision::SuppressedTransientSimilarity
    ) || score.forward_adjusted_cost >= score.threshold
}

fn mark_forward_similarity_suppressed(
    scores: &mut BTreeMap<usize, ScenecutResult>,
    frame: usize,
    return_frame: usize,
    return_candidate_frame: usize,
    delta: Option<f64>,
) {
    let Some(score) = scores.get_mut(&frame) else {
        return;
    };

    score.decision = ScenecutDecision::SuppressedForwardSimilarity;
    score.forward_return_frame = Some(return_frame);
    if let Some(delta) = delta {
        score.forward_similarity_score = Some(delta);
    }

    for candidate in score.forward_similarity_candidates.iter_mut().flatten() {
        if candidate.frame == return_frame {
            candidate.decision = ForwardSimilarityCandidateDecision::Accepted;
            candidate.candidate_frame = Some(return_candidate_frame);
            candidate.candidate_offset = Some(return_candidate_frame - frame + 1);
            break;
        }
    }
}

const TEXT_CARD_CLUSTER_MIN_CUTS: usize = 3;
const TEXT_CARD_CLUSTER_MAX_GAP: usize = 130;
const TEXT_CARD_MAX_COST_RATIO: f64 = 0.40;
const TEXT_CARD_MAX_LUMA_8BIT: f64 = 35.0;
const TEXT_CARD_MAX_IMP_RATIO: f64 = 4.0;
const TEXT_CARD_MIN_GLOBAL_IMP_RATIO: f64 = 3.8;
const TEXT_CARD_MAX_SIMILARITY_DELTA_8BIT: f64 = 6.0;
const TEXT_CARD_MAX_TRANSIENT_DELTA_8BIT: f64 = 3.0;

fn apply_text_card_cluster_postprocess(
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let cuts = keyframes
        .iter()
        .copied()
        .filter(|&frame| frame != 0)
        .collect::<Vec<_>>();
    let mut run = Vec::new();

    for frame in cuts {
        let weak_text_card_cut = scores.get(&frame).copied().is_some_and(weak_text_card_cut);
        let extends_run = run
            .last()
            .is_none_or(|previous| frame - previous <= TEXT_CARD_CLUSTER_MAX_GAP);
        if weak_text_card_cut && extends_run {
            run.push(frame);
        } else {
            suppress_text_card_run(&run, keyframes, scores);
            run.clear();
            if weak_text_card_cut {
                run.push(frame);
            }
        }
    }

    suppress_text_card_run(&run, keyframes, scores);
}

fn suppress_text_card_run(
    run: &[usize],
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    if run.len() < TEXT_CARD_CLUSTER_MIN_CUTS {
        return;
    }

    for &frame in run {
        keyframes.remove(&frame);
        if let Some(score) = scores.get_mut(&frame) {
            score.decision = ScenecutDecision::SuppressedTextCardCluster;
        }
    }
}

fn weak_text_card_cut(score: ScenecutResult) -> bool {
    if score.decision != ScenecutDecision::CutImportance {
        return false;
    }
    if score.cost_ratio > TEXT_CARD_MAX_COST_RATIO
        || score.avg_luma_8bit > TEXT_CARD_MAX_LUMA_8BIT
        || score.imp_block_ratio > TEXT_CARD_MAX_IMP_RATIO
        || score.global_imp_block_ratio < TEXT_CARD_MIN_GLOBAL_IMP_RATIO
    {
        return false;
    }

    score
        .transient_similarity_score
        .is_some_and(|delta| delta <= TEXT_CARD_MAX_TRANSIENT_DELTA_8BIT)
        || score
            .forward_similarity_candidates
            .iter()
            .flatten()
            .any(|candidate| candidate.delta <= TEXT_CARD_MAX_SIMILARITY_DELTA_8BIT)
}

const FAST_MOTION_PAIR_MAX_GAP: usize = 24;
const FAST_MOTION_PAIR_MAX_EXIT_GAP: usize = 80;
const FAST_MOTION_PAIR_MIN_PREVIOUS_GAP: usize = 80;
const FAST_MOTION_MAX_COST_RATIO: f64 = 0.35;
const FAST_MOTION_MIN_GLOBAL_IMP_RATIO: f64 = 4.0;
const FAST_MOTION_MIN_IMP_RATIO: f64 = 2.7;
const FAST_MOTION_MIN_LUMA_8BIT: f64 = 45.0;
const FAST_MOTION_MIN_TRANSIENT_DELTA_8BIT: f64 = 6.0;
const FAST_MOTION_EXIT_MIN_COST_RATIO: f64 = 1.2;

fn apply_fast_motion_micro_split_postprocess(
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let cuts = keyframes
        .iter()
        .copied()
        .filter(|&frame| frame != 0)
        .collect::<Vec<_>>();

    for window in cuts.windows(4) {
        let &[previous, first, second, exit] = window else {
            continue;
        };
        if first - previous < FAST_MOTION_PAIR_MIN_PREVIOUS_GAP
            || second - first > FAST_MOTION_PAIR_MAX_GAP
            || exit - second > FAST_MOTION_PAIR_MAX_EXIT_GAP
        {
            continue;
        }
        let Some(first_score) = scores.get(&first).copied() else {
            continue;
        };
        let Some(second_score) = scores.get(&second).copied() else {
            continue;
        };
        let Some(exit_score) = scores.get(&exit).copied() else {
            continue;
        };
        if !weak_fast_motion_split(first_score)
            || !weak_fast_motion_split(second_score)
            || !strong_fast_motion_exit(exit_score)
        {
            continue;
        }

        for frame in [first, second] {
            keyframes.remove(&frame);
            if let Some(score) = scores.get_mut(&frame) {
                score.decision = ScenecutDecision::SuppressedFastMotion;
            }
        }
    }
}

fn weak_fast_motion_split(score: ScenecutResult) -> bool {
    score.decision == ScenecutDecision::CutImportance
        && score.cost_ratio <= FAST_MOTION_MAX_COST_RATIO
        && score.imp_block_ratio >= FAST_MOTION_MIN_IMP_RATIO
        && score.global_imp_block_ratio >= FAST_MOTION_MIN_GLOBAL_IMP_RATIO
        && score.avg_luma_8bit >= FAST_MOTION_MIN_LUMA_8BIT
        && score
            .transient_similarity_score
            .is_some_and(|delta| delta >= FAST_MOTION_MIN_TRANSIENT_DELTA_8BIT)
}

fn strong_fast_motion_exit(score: ScenecutResult) -> bool {
    score.decision == ScenecutDecision::Cut && score.cost_ratio >= FAST_MOTION_EXIT_MIN_COST_RATIO
}

const DARK_OCCLUSION_MAX_COST_RATIO: f64 = 0.25;
const DARK_OCCLUSION_MAX_LUMA_8BIT: f64 = 25.0;
const DARK_OCCLUSION_MIN_IMP_RATIO: f64 = 3.5;
const DARK_OCCLUSION_MIN_GLOBAL_IMP_RATIO: f64 = 4.5;
const DARK_OCCLUSION_MIN_BAD_BLOCK_RATIO: f64 = 0.70;
const DARK_OCCLUSION_MAX_GOOD_BLOCK_RATIO: f64 = 0.002;
const DARK_OCCLUSION_MAX_TRANSIENT_DELTA_8BIT: f64 = 4.0;
const DARK_OCCLUSION_MAX_FORWARD_DELTA_8BIT: f64 = 5.0;

fn apply_dark_occlusion_postprocess(
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let cuts = keyframes
        .iter()
        .copied()
        .filter(|&frame| frame != 0)
        .collect::<Vec<_>>();
    for frame in cuts {
        let suppress = scores.get(&frame).copied().is_some_and(dark_occlusion_cut);
        if !suppress {
            continue;
        }
        keyframes.remove(&frame);
        if let Some(score) = scores.get_mut(&frame) {
            score.decision = ScenecutDecision::SuppressedDarkOcclusion;
        }
    }
}

fn dark_occlusion_cut(score: ScenecutResult) -> bool {
    score.decision == ScenecutDecision::CutImportance
        && score.cost_ratio <= DARK_OCCLUSION_MAX_COST_RATIO
        && score.avg_luma_8bit <= DARK_OCCLUSION_MAX_LUMA_8BIT
        && score.imp_block_ratio >= DARK_OCCLUSION_MIN_IMP_RATIO
        && score.global_imp_block_ratio >= DARK_OCCLUSION_MIN_GLOBAL_IMP_RATIO
        && score.static_bad_block_ratio >= DARK_OCCLUSION_MIN_BAD_BLOCK_RATIO
        && score.static_good_block_ratio <= DARK_OCCLUSION_MAX_GOOD_BLOCK_RATIO
        && score
            .transient_similarity_score
            .is_some_and(|delta| delta <= DARK_OCCLUSION_MAX_TRANSIENT_DELTA_8BIT)
        && score
            .forward_similarity_candidates
            .iter()
            .flatten()
            .any(|candidate| candidate.delta <= DARK_OCCLUSION_MAX_FORWARD_DELTA_8BIT)
}

const DARK_SCENE_PEAK_MIN_SCENE_LEN: usize = 120;
const DARK_SCENE_PEAK_LOCAL_RADIUS: usize = 5;
const DARK_SCENE_PEAK_DENSITY_RADIUS: usize = 500;
const DARK_SCENE_PEAK_WIDE_DENSITY_RADIUS: usize = 2000;
const DARK_SCENE_PEAK_MAX_LOCAL_CUTS: usize = 10;
const DARK_SCENE_PEAK_MAX_WIDE_CUTS: usize = 35;
const DARK_SCENE_PEAK_MAX_ADJACENT_COST_RATIO: f64 = 5.0;
const DARK_SCENE_PEAK_MIN_COST_RATIO: f64 = 0.25;
const DARK_SCENE_PEAK_MAX_COST_RATIO: f64 = 0.45;
const DARK_SCENE_PEAK_MAX_LUMA_8BIT: f64 = 50.0;
const DARK_SCENE_PEAK_MIN_IMP_RATIO: f64 = 1.3;
const DARK_SCENE_PEAK_MIN_GLOBAL_IMP_RATIO: f64 = 1.4;
const DARK_SCENE_PEAK_MIN_BAD_BLOCK_RATIO: f64 = 0.62;
const DARK_SCENE_PEAK_MAX_GOOD_BLOCK_RATIO: f64 = 0.005;

fn apply_dark_scene_peak_recovery_postprocess(
    opts: DetectionOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let min_distance = opts.min_scenecut_distance.unwrap_or(0);
    let candidates = scores
        .iter()
        .filter_map(|(&frame, &score)| {
            dark_scene_peak_recovery_candidate(frame, score, keyframes, scores, min_distance)
                .then_some(frame)
        })
        .collect::<Vec<_>>();

    for frame in candidates {
        if !scene_distance_ok(frame, keyframes, min_distance) {
            continue;
        }
        keyframes.insert(frame);
        if let Some(score) = scores.get_mut(&frame) {
            score.decision = ScenecutDecision::CutDarkScenePeak;
        }
    }
}

fn dark_scene_peak_recovery_candidate(
    frame: usize,
    score: ScenecutResult,
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
    min_distance: usize,
) -> bool {
    if score.decision != ScenecutDecision::NoCut {
        return false;
    }
    if !dark_scene_peak_score(score)
        || !local_cost_peak(frame, scores, DARK_SCENE_PEAK_LOCAL_RADIUS)
    {
        return false;
    }
    if local_cut_count(keyframes, frame, DARK_SCENE_PEAK_DENSITY_RADIUS)
        > DARK_SCENE_PEAK_MAX_LOCAL_CUTS
        || local_cut_count(keyframes, frame, DARK_SCENE_PEAK_WIDE_DENSITY_RADIUS)
            > DARK_SCENE_PEAK_MAX_WIDE_CUTS
    {
        return false;
    }

    let Some(previous_cut) = keyframes.range(..frame).next_back().copied() else {
        return false;
    };
    let Some(next_cut) = keyframes.range((frame + 1)..).next().copied() else {
        return false;
    };
    if next_cut - previous_cut < DARK_SCENE_PEAK_MIN_SCENE_LEN
        || frame - previous_cut < min_distance
        || next_cut - frame < min_distance
    {
        return false;
    }

    let previous_cost_ratio = scores
        .get(&previous_cut)
        .map_or(0.0, |score| score.cost_ratio);
    let next_cost_ratio = scores.get(&next_cut).map_or(0.0, |score| score.cost_ratio);
    previous_cost_ratio <= DARK_SCENE_PEAK_MAX_ADJACENT_COST_RATIO
        && next_cost_ratio <= DARK_SCENE_PEAK_MAX_ADJACENT_COST_RATIO
}

fn dark_scene_peak_score(score: ScenecutResult) -> bool {
    score.cost_ratio >= DARK_SCENE_PEAK_MIN_COST_RATIO
        && score.cost_ratio <= DARK_SCENE_PEAK_MAX_COST_RATIO
        && score.avg_luma_8bit <= DARK_SCENE_PEAK_MAX_LUMA_8BIT
        && score.imp_block_ratio >= DARK_SCENE_PEAK_MIN_IMP_RATIO
        && score.global_imp_block_ratio >= DARK_SCENE_PEAK_MIN_GLOBAL_IMP_RATIO
        && score.static_bad_block_ratio >= DARK_SCENE_PEAK_MIN_BAD_BLOCK_RATIO
        && score.static_good_block_ratio <= DARK_SCENE_PEAK_MAX_GOOD_BLOCK_RATIO
}

fn local_cost_peak(frame: usize, scores: &BTreeMap<usize, ScenecutResult>, radius: usize) -> bool {
    let Some(score) = scores.get(&frame) else {
        return false;
    };
    let start = frame.saturating_sub(radius);
    let end = frame.saturating_add(radius);
    scores
        .range(start..=end)
        .all(|(&other_frame, other_score)| {
            other_frame == frame || other_score.cost_ratio <= score.cost_ratio
        })
}

fn local_cut_count(keyframes: &BTreeSet<usize>, frame: usize, radius: usize) -> usize {
    let start = frame.saturating_sub(radius);
    let end = frame.saturating_add(radius);
    keyframes
        .range(start..=end)
        .filter(|&&frame| frame != 0)
        .count()
}

fn scene_distance_ok(frame: usize, keyframes: &BTreeSet<usize>, min_distance: usize) -> bool {
    let previous_ok = keyframes
        .range(..frame)
        .next_back()
        .is_none_or(|&previous| frame - previous >= min_distance);
    let next_ok = keyframes
        .range((frame + 1)..)
        .next()
        .is_none_or(|&next| next - frame >= min_distance);
    previous_ok && next_ok
}

const SPARSE_SCENE_PEAK_MIN_SCENE_LEN: usize = 120;
const SPARSE_SCENE_PEAK_LOCAL_RADIUS: usize = 5;
const SPARSE_SCENE_PEAK_DENSITY_RADIUS: usize = 500;
const SPARSE_SCENE_PEAK_WIDE_DENSITY_RADIUS: usize = 2000;
const SPARSE_SCENE_PEAK_MAX_LOCAL_CUTS: usize = 10;
const SPARSE_SCENE_PEAK_MAX_WIDE_CUTS: usize = 35;
const SPARSE_SCENE_PEAK_MAX_ADJACENT_COST_RATIO: f64 = 5.0;
const SPARSE_SCENE_PEAK_MIN_COST_RATIO: f64 = 0.25;
const SPARSE_SCENE_PEAK_MAX_COST_RATIO: f64 = 1.0;
const SPARSE_SCENE_PEAK_MIN_IMP_RATIO: f64 = 1.5;
const SPARSE_SCENE_PEAK_MIN_GLOBAL_IMP_RATIO: f64 = 1.8;
const SPARSE_SCENE_PEAK_MAX_LUMA_8BIT: f64 = 90.0;
const SPARSE_SCENE_PEAK_MIN_BAD_BLOCK_RATIO: f64 = 0.60;
const SPARSE_SCENE_PEAK_MAX_GOOD_BLOCK_RATIO: f64 = 0.10;

fn apply_sparse_scene_peak_recovery_postprocess(
    opts: DetectionOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let min_distance = opts.min_scenecut_distance.unwrap_or(0);
    let candidates = scores
        .iter()
        .filter_map(|(&frame, &score)| {
            sparse_scene_peak_recovery_candidate(frame, score, keyframes, scores, min_distance)
                .then_some(frame)
        })
        .collect::<Vec<_>>();

    for frame in candidates {
        if !scene_distance_ok(frame, keyframes, min_distance) {
            continue;
        }
        keyframes.insert(frame);
        if let Some(score) = scores.get_mut(&frame) {
            score.decision = ScenecutDecision::CutSparseScenePeak;
        }
    }
}

fn sparse_scene_peak_recovery_candidate(
    frame: usize,
    score: ScenecutResult,
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
    min_distance: usize,
) -> bool {
    if score.decision != ScenecutDecision::NoCut {
        return false;
    }
    if !sparse_scene_peak_score(score)
        || !local_cost_peak(frame, scores, SPARSE_SCENE_PEAK_LOCAL_RADIUS)
    {
        return false;
    }
    if local_cut_count(keyframes, frame, SPARSE_SCENE_PEAK_DENSITY_RADIUS)
        > SPARSE_SCENE_PEAK_MAX_LOCAL_CUTS
        || local_cut_count(keyframes, frame, SPARSE_SCENE_PEAK_WIDE_DENSITY_RADIUS)
            > SPARSE_SCENE_PEAK_MAX_WIDE_CUTS
    {
        return false;
    }

    let Some(previous_cut) = keyframes.range(..frame).next_back().copied() else {
        return false;
    };
    let Some(next_cut) = keyframes.range((frame + 1)..).next().copied() else {
        return false;
    };
    if next_cut - previous_cut < SPARSE_SCENE_PEAK_MIN_SCENE_LEN
        || frame - previous_cut < min_distance
        || next_cut - frame < min_distance
    {
        return false;
    }

    let previous_cost_ratio = scores
        .get(&previous_cut)
        .map_or(0.0, |score| score.cost_ratio);
    let next_cost_ratio = scores.get(&next_cut).map_or(0.0, |score| score.cost_ratio);
    previous_cost_ratio <= SPARSE_SCENE_PEAK_MAX_ADJACENT_COST_RATIO
        && next_cost_ratio <= SPARSE_SCENE_PEAK_MAX_ADJACENT_COST_RATIO
}

fn sparse_scene_peak_score(score: ScenecutResult) -> bool {
    score.cost_ratio >= SPARSE_SCENE_PEAK_MIN_COST_RATIO
        && score.cost_ratio <= SPARSE_SCENE_PEAK_MAX_COST_RATIO
        && score.imp_block_ratio >= SPARSE_SCENE_PEAK_MIN_IMP_RATIO
        && score.global_imp_block_ratio >= SPARSE_SCENE_PEAK_MIN_GLOBAL_IMP_RATIO
        && score.avg_luma_8bit <= SPARSE_SCENE_PEAK_MAX_LUMA_8BIT
        && score.static_bad_block_ratio >= SPARSE_SCENE_PEAK_MIN_BAD_BLOCK_RATIO
        && score.static_good_block_ratio <= SPARSE_SCENE_PEAK_MAX_GOOD_BLOCK_RATIO
}

#[derive(Clone, Copy)]
struct RefinedSparsePeakShift {
    from: usize,
    to: usize,
}

const REFINED_SPARSE_PEAK_MIN_DISTANCE: usize = 40;
const REFINED_SPARSE_PEAK_SHIFT_MAX_DISTANCE: usize = 40;
const REFINED_SPARSE_PEAK_MIN_COST_RATIO: f64 = 0.65;
const REFINED_SPARSE_PEAK_MAX_COST_RATIO: f64 = 1.05;
const REFINED_SPARSE_PEAK_MIN_IMP_RATIO: f64 = 3.0;
const REFINED_SPARSE_PEAK_MIN_GLOBAL_IMP_RATIO: f64 = 3.5;
const REFINED_SPARSE_PEAK_MIN_LUMA_8BIT: f64 = 55.0;
const REFINED_SPARSE_PEAK_MAX_LUMA_8BIT: f64 = 90.0;
const REFINED_SPARSE_PEAK_MIN_BAD_BLOCK_RATIO: f64 = 0.50;
const REFINED_SPARSE_PEAK_MAX_GOOD_BLOCK_RATIO: f64 = 0.09;
const REFINED_SPARSE_PEAK_MIN_EDGE_DELTA_8BIT: f64 = 10.0;
const REFINED_SPARSE_PEAK_MAX_REPEAT_DELTA_8BIT: f64 = 5.0;
const REFINED_SPARSE_PEAK_MIN_DISTINCT_DELTA_8BIT: f64 = 8.0;
const REFINED_SPARSE_PEAK_DENSITY_RADIUS: usize = 500;
const REFINED_SPARSE_PEAK_WIDE_DENSITY_RADIUS: usize = 2000;
const REFINED_SPARSE_PEAK_MAX_LOCAL_CUTS: usize = 12;
const REFINED_SPARSE_PEAK_MAX_WIDE_CUTS: usize = 40;
const REFINED_SPARSE_PEAK_MAX_ADJACENT_COST_RATIO: f64 = 5.0;
const REFINED_SPARSE_PEAK_ABA_MAX_SEGMENT_LEN: usize = 130;

fn apply_refined_sparse_peak_postprocess(
    opts: DetectionOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let min_distance = opts
        .min_scenecut_distance
        .unwrap_or(0)
        .max(REFINED_SPARSE_PEAK_MIN_DISTANCE);
    let base_min_distance = opts.min_scenecut_distance.unwrap_or(0);

    let shifts = refined_sparse_peak_shifts(base_min_distance, keyframes, scores);
    let shifted_to = shifts.iter().map(|shift| shift.to).collect::<BTreeSet<_>>();
    for shift in shifts {
        keyframes.remove(&shift.from);
        if let Some(score) = scores.get_mut(&shift.from) {
            score.decision = ScenecutDecision::SuppressedRefinedSparsePeak;
        }
        keyframes.insert(shift.to);
        if let Some(score) = scores.get_mut(&shift.to) {
            score.decision = ScenecutDecision::CutRefinedSparsePeak;
        }
    }

    let candidates = scores
        .iter()
        .filter_map(|(&frame, &score)| {
            refined_sparse_peak_recovery_candidate(frame, score, keyframes, scores, min_distance)
                .then_some(frame)
        })
        .collect::<Vec<_>>();

    let mut recovered = shifted_to;
    for frame in candidates {
        if keyframes.contains(&frame) || !scene_distance_ok(frame, keyframes, min_distance) {
            continue;
        }
        keyframes.insert(frame);
        recovered.insert(frame);
        if let Some(score) = scores.get_mut(&frame) {
            score.decision = ScenecutDecision::CutRefinedSparsePeak;
        }
    }

    let suppress = refined_sparse_peak_aba_suppressions(
        base_min_distance,
        min_distance,
        keyframes,
        scores,
        &recovered,
    );
    for frame in suppress {
        keyframes.remove(&frame);
        if let Some(score) = scores.get_mut(&frame) {
            score.decision = ScenecutDecision::SuppressedRefinedSparsePeak;
        }
    }
}

fn refined_sparse_peak_shifts(
    min_distance: usize,
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
) -> Vec<RefinedSparsePeakShift> {
    let min_distance = min_distance.max(1);
    keyframes
        .iter()
        .copied()
        .filter_map(|frame| {
            let score = scores.get(&frame).copied()?;
            if score.decision != ScenecutDecision::CutSparseScenePeak {
                return None;
            }
            let next_cut = keyframes.range((frame + 1)..).next().copied()?;
            let next_next_cut = keyframes.range((next_cut + 1)..).next().copied()?;
            if frame_signature_delta_for_scores(frame, next_next_cut, scores)?
                > REFINED_SPARSE_PEAK_MAX_REPEAT_DELTA_8BIT
            {
                return None;
            }

            let search_end = frame
                .saturating_add(REFINED_SPARSE_PEAK_SHIFT_MAX_DISTANCE)
                .min(next_cut.saturating_sub(min_distance));
            ((frame + min_distance)..=search_end)
                .filter(|&candidate| {
                    let Some(candidate_score) = scores.get(&candidate).copied() else {
                        return false;
                    };
                    if !refined_sparse_peak_score(candidate, candidate_score, scores) {
                        return false;
                    }
                    let Some(candidate_next_next_delta) =
                        frame_signature_delta_for_scores(candidate, next_next_cut, scores)
                    else {
                        return false;
                    };
                    let Some(candidate_from_delta) =
                        frame_signature_delta_for_scores(frame, candidate, scores)
                    else {
                        return false;
                    };
                    candidate_next_next_delta > REFINED_SPARSE_PEAK_MIN_DISTINCT_DELTA_8BIT
                        && candidate_from_delta >= REFINED_SPARSE_PEAK_MIN_DISTINCT_DELTA_8BIT
                })
                .max_by(|&left, &right| {
                    refined_sparse_peak_rank(left, scores)
                        .partial_cmp(&refined_sparse_peak_rank(right, scores))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|to| RefinedSparsePeakShift { from: frame, to })
        })
        .collect()
}

fn refined_sparse_peak_recovery_candidate(
    frame: usize,
    score: ScenecutResult,
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
    min_distance: usize,
) -> bool {
    if keyframes.contains(&frame) || !refined_sparse_peak_score(frame, score, scores) {
        return false;
    }
    if local_cut_count(keyframes, frame, REFINED_SPARSE_PEAK_DENSITY_RADIUS)
        > REFINED_SPARSE_PEAK_MAX_LOCAL_CUTS
        || local_cut_count(keyframes, frame, REFINED_SPARSE_PEAK_WIDE_DENSITY_RADIUS)
            > REFINED_SPARSE_PEAK_MAX_WIDE_CUTS
    {
        return false;
    }

    let Some(previous_cut) = keyframes.range(..frame).next_back().copied() else {
        return false;
    };
    let Some(next_cut) = keyframes.range((frame + 1)..).next().copied() else {
        return false;
    };
    if next_cut - previous_cut < SPARSE_SCENE_PEAK_MIN_SCENE_LEN
        || frame - previous_cut < min_distance
        || next_cut - frame < min_distance
        || refined_sparse_repeat_context(previous_cut, frame, next_cut, scores)
    {
        return false;
    }

    let previous_cost_ratio = scores
        .get(&previous_cut)
        .map_or(0.0, |score| score.cost_ratio);
    let next_cost_ratio = scores.get(&next_cut).map_or(0.0, |score| score.cost_ratio);
    previous_cost_ratio <= REFINED_SPARSE_PEAK_MAX_ADJACENT_COST_RATIO
        && next_cost_ratio <= REFINED_SPARSE_PEAK_MAX_ADJACENT_COST_RATIO
}

fn refined_sparse_peak_score(
    frame: usize,
    score: ScenecutResult,
    scores: &BTreeMap<usize, ScenecutResult>,
) -> bool {
    score.decision == ScenecutDecision::NoCut
        && score.cost_ratio >= REFINED_SPARSE_PEAK_MIN_COST_RATIO
        && score.cost_ratio <= REFINED_SPARSE_PEAK_MAX_COST_RATIO
        && score.imp_block_ratio >= REFINED_SPARSE_PEAK_MIN_IMP_RATIO
        && score.global_imp_block_ratio >= REFINED_SPARSE_PEAK_MIN_GLOBAL_IMP_RATIO
        && score.avg_luma_8bit >= REFINED_SPARSE_PEAK_MIN_LUMA_8BIT
        && score.avg_luma_8bit <= REFINED_SPARSE_PEAK_MAX_LUMA_8BIT
        && score.static_bad_block_ratio >= REFINED_SPARSE_PEAK_MIN_BAD_BLOCK_RATIO
        && score.static_good_block_ratio <= REFINED_SPARSE_PEAK_MAX_GOOD_BLOCK_RATIO
        && frame_signature_delta_for_scores(frame.saturating_sub(1), frame, scores)
            .is_some_and(|delta| delta >= REFINED_SPARSE_PEAK_MIN_EDGE_DELTA_8BIT)
        && local_cost_peak(frame, scores, SPARSE_SCENE_PEAK_LOCAL_RADIUS)
}

fn refined_sparse_peak_rank(frame: usize, scores: &BTreeMap<usize, ScenecutResult>) -> f64 {
    scores.get(&frame).map_or(0.0, |score| {
        score.cost_ratio + 0.05 * score.imp_block_ratio + 0.03 * score.global_imp_block_ratio
    })
}

fn refined_sparse_repeat_context(
    previous_cut: usize,
    frame: usize,
    next_cut: usize,
    scores: &BTreeMap<usize, ScenecutResult>,
) -> bool {
    [
        frame_signature_delta_for_scores(previous_cut, next_cut, scores),
        frame_signature_delta_for_scores(previous_cut, frame, scores),
        frame_signature_delta_for_scores(frame, next_cut, scores),
    ]
    .into_iter()
    .flatten()
    .any(|delta| delta <= REFINED_SPARSE_PEAK_MAX_REPEAT_DELTA_8BIT)
}

fn refined_sparse_peak_aba_suppressions(
    base_min_distance: usize,
    min_distance: usize,
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
    recovered: &BTreeSet<usize>,
) -> Vec<usize> {
    keyframes
        .iter()
        .copied()
        .filter(|&frame| {
            let Some(score) = scores.get(&frame).copied() else {
                return false;
            };
            if score.decision != ScenecutDecision::CutSparseScenePeak {
                return false;
            }
            let Some(previous_cut) = keyframes.range(..frame).next_back().copied() else {
                return false;
            };
            if !recovered.contains(&previous_cut)
                || frame - previous_cut > REFINED_SPARSE_PEAK_ABA_MAX_SEGMENT_LEN
            {
                return false;
            }
            let Some(next_cut) = keyframes.range((frame + 1)..).next().copied() else {
                return false;
            };
            if next_cut - frame > REFINED_SPARSE_PEAK_ABA_MAX_SEGMENT_LEN {
                return false;
            }

            let search_start = frame.saturating_add(base_min_distance.max(1));
            let search_end = next_cut.saturating_sub(base_min_distance.max(1));
            search_start <= search_end
                && (search_start..=search_end).any(|candidate| {
                    candidate - frame < min_distance
                        && scores
                            .get(&candidate)
                            .copied()
                            .is_some_and(|candidate_score| {
                                refined_sparse_peak_score(candidate, candidate_score, scores)
                            })
                })
        })
        .collect()
}

fn frame_signature_delta_for_scores(
    left: usize,
    right: usize,
    scores: &BTreeMap<usize, ScenecutResult>,
) -> Option<f64> {
    Some(frame_signature_delta_8bit(
        scores.get(&left)?.frame_luma_signature?,
        scores.get(&right)?.frame_luma_signature?,
    ))
}

#[derive(Clone, Copy)]
struct AbaReturnCandidate {
    frame: usize,
    previous_cut: usize,
    next_cut: usize,
    return_delta: f64,
    edge_delta: f64,
    previous_edge_delta: f64,
}

const ABA_RETURN_MIN_COST_RATIO: f64 = 0.25;
const ABA_RETURN_MAX_COST_RATIO: f64 = 1.05;
const ABA_RETURN_MIN_IMP_RATIO: f64 = 2.0;
const ABA_RETURN_MIN_GLOBAL_IMP_RATIO: f64 = 3.0;
const ABA_RETURN_MAX_LUMA_8BIT: f64 = 90.0;
const ABA_RETURN_MAX_GOOD_BLOCK_RATIO: f64 = 0.10;
const ABA_RETURN_MAX_SIGNATURE_DELTA_8BIT: f64 = 4.0;
const ABA_RETURN_MIN_EDGE_DELTA_8BIT: f64 = 5.0;
const ABA_RETURN_MIN_EDGE_RATIO: f64 = 2.0;

fn apply_aba_return_recovery_postprocess(
    opts: DetectionOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let min_distance = opts.min_scenecut_distance.unwrap_or(0);
    let mut candidates = scores
        .iter()
        .filter_map(|(&frame, &score)| {
            aba_return_recovery_candidate(frame, score, keyframes, scores, min_distance)
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|left, right| {
        left.previous_cut
            .cmp(&right.previous_cut)
            .then(left.next_cut.cmp(&right.next_cut))
            .then_with(|| {
                left.return_delta
                    .partial_cmp(&right.return_delta)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                right
                    .edge_delta
                    .partial_cmp(&left.edge_delta)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                right
                    .previous_edge_delta
                    .partial_cmp(&left.previous_edge_delta)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let mut selected_intervals = BTreeSet::new();
    for candidate in candidates {
        if !selected_intervals.insert((candidate.previous_cut, candidate.next_cut)) {
            continue;
        }
        if !scene_distance_ok(candidate.frame, keyframes, min_distance) {
            continue;
        }
        keyframes.insert(candidate.frame);
        if let Some(score) = scores.get_mut(&candidate.frame) {
            score.decision = ScenecutDecision::CutAbaReturn;
        }
    }
}

fn aba_return_recovery_candidate(
    frame: usize,
    score: ScenecutResult,
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
    min_distance: usize,
) -> Option<AbaReturnCandidate> {
    if keyframes.contains(&frame) || !aba_return_score(score) {
        return None;
    }
    let previous_cut = keyframes.range(..frame).next_back().copied()?;
    let next_cut = keyframes.range((frame + 1)..).next().copied()?;
    if previous_cut == 0
        || frame <= previous_cut
        || frame - previous_cut < min_distance
        || next_cut - frame < min_distance
    {
        return None;
    }

    let previous_end = scores
        .get(&previous_cut.saturating_sub(1))?
        .frame_luma_signature?;
    let previous_start = scores.get(&previous_cut)?.frame_luma_signature?;
    let before_frame = scores.get(&frame.saturating_sub(1))?.frame_luma_signature?;
    let current = score.frame_luma_signature?;
    let return_delta = frame_signature_delta_8bit(previous_end, current);
    let edge_delta = frame_signature_delta_8bit(before_frame, current);
    let previous_edge_delta = frame_signature_delta_8bit(previous_end, previous_start);
    let edge_ratio = edge_delta.min(previous_edge_delta) / return_delta.max(0.001);

    if return_delta > ABA_RETURN_MAX_SIGNATURE_DELTA_8BIT
        || edge_delta < ABA_RETURN_MIN_EDGE_DELTA_8BIT
        || previous_edge_delta < ABA_RETURN_MIN_EDGE_DELTA_8BIT
        || edge_ratio < ABA_RETURN_MIN_EDGE_RATIO
    {
        return None;
    }

    Some(AbaReturnCandidate {
        frame,
        previous_cut,
        next_cut,
        return_delta,
        edge_delta,
        previous_edge_delta,
    })
}

fn aba_return_score(score: ScenecutResult) -> bool {
    matches!(
        score.decision,
        ScenecutDecision::NoCut | ScenecutDecision::SuppressedTransientSimilarity
    ) && score.cost_ratio >= ABA_RETURN_MIN_COST_RATIO
        && score.cost_ratio <= ABA_RETURN_MAX_COST_RATIO
        && score.imp_block_ratio >= ABA_RETURN_MIN_IMP_RATIO
        && score.global_imp_block_ratio >= ABA_RETURN_MIN_GLOBAL_IMP_RATIO
        && score.avg_luma_8bit <= ABA_RETURN_MAX_LUMA_8BIT
        && score.static_good_block_ratio <= ABA_RETURN_MAX_GOOD_BLOCK_RATIO
}

fn frame_signature_delta_8bit(
    left: [u8; analyze::FRAME_LUMA_SIGNATURE_CELLS],
    right: [u8; analyze::FRAME_LUMA_SIGNATURE_CELLS],
) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&left, right)| u32::from(left.abs_diff(right)))
        .sum::<u32>() as f64
        / analyze::FRAME_LUMA_SIGNATURE_CELLS as f64
}

#[derive(Clone)]
struct AbaChainCandidate {
    boundaries: Vec<usize>,
    suppress: Vec<usize>,
}

const ABA_CHAIN_MIN_SEGMENTS: usize = 4;
const ABA_CHAIN_MAX_SIGNATURE_DELTA_8BIT: f64 = 5.0;
const ABA_CHAIN_MIN_DIFFERENT_DELTA_8BIT: f64 = 8.0;
const ABA_CHAIN_MAX_SEGMENT_EXTRA: usize = 50;
const ABA_CHAIN_MIN_HIDDEN_COST_RATIO: f64 = 0.25;
const ABA_CHAIN_MAX_HIDDEN_COST_RATIO: f64 = 1.2;
const ABA_CHAIN_MIN_HIDDEN_IMP_RATIO: f64 = 2.0;
const ABA_CHAIN_MIN_HIDDEN_GLOBAL_IMP_RATIO: f64 = 3.0;
const ABA_CHAIN_MAX_HIDDEN_LUMA_8BIT: f64 = 95.0;
const ABA_CHAIN_MAX_HIDDEN_GOOD_BLOCK_RATIO: f64 = 0.12;
const ABA_CHAIN_MIN_HIDDEN_EDGE_DELTA_8BIT: f64 = 5.0;

fn apply_aba_chain_compaction_postprocess(
    opts: DetectionOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let break_segment_len = opts.tuning.forward_similarity.frames.max(1);
    let max_segment_len = break_segment_len
        .saturating_add(break_segment_len / 2)
        .saturating_add(ABA_CHAIN_MAX_SEGMENT_EXTRA.min(break_segment_len));
    let boundaries = aba_chain_boundaries(keyframes, scores);
    let mut chains = Vec::new();

    for start_idx in 0..boundaries.len().saturating_sub(2) {
        if let Some(chain) = aba_chain_from_boundary(
            start_idx,
            &boundaries,
            keyframes,
            scores,
            break_segment_len,
            max_segment_len,
        ) {
            chains.push(chain);
        }
    }

    chains.sort_by(|left, right| {
        (right.boundaries.len() - 1)
            .cmp(&(left.boundaries.len() - 1))
            .then(left.boundaries[0].cmp(&right.boundaries[0]))
    });

    let mut suppress = BTreeSet::new();
    for chain in chains {
        if chain
            .boundaries
            .iter()
            .any(|frame| suppress.contains(frame))
        {
            continue;
        }
        suppress.extend(chain.suppress);
    }

    for frame in suppress {
        keyframes.remove(&frame);
        if let Some(score) = scores.get_mut(&frame) {
            score.decision = ScenecutDecision::SuppressedAbaChain;
        }
    }
}

fn aba_chain_boundaries(
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
) -> Vec<usize> {
    keyframes
        .iter()
        .copied()
        .filter(|&frame| frame != 0)
        .chain(scores.iter().filter_map(|(&frame, &score)| {
            aba_chain_hidden_boundary(frame, score, keyframes, scores).then_some(frame)
        }))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn aba_chain_from_boundary(
    start_idx: usize,
    boundaries: &[usize],
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
    break_segment_len: usize,
    max_segment_len: usize,
) -> Option<AbaChainCandidate> {
    let first = boundaries[start_idx];
    let second = boundaries.get(start_idx + 1).copied()?;
    if second - first > max_segment_len
        || aba_chain_signature_delta(first, second, scores)? < ABA_CHAIN_MIN_DIFFERENT_DELTA_8BIT
    {
        return None;
    }

    let mut chain = vec![first, second];
    for &candidate in &boundaries[start_idx + 2..] {
        let previous = *chain.last()?;
        let previous_segment_idx = chain.len().saturating_sub(2);
        if previous_segment_idx >= 1 && previous - chain[chain.len() - 2] > break_segment_len {
            break;
        }
        if candidate - previous > max_segment_len {
            break;
        }

        let Some(same_as_previous_tag) =
            aba_chain_signature_delta(chain[chain.len() - 2], candidate, scores)
        else {
            break;
        };
        let Some(different_from_previous) = aba_chain_signature_delta(previous, candidate, scores)
        else {
            break;
        };
        if same_as_previous_tag <= ABA_CHAIN_MAX_SIGNATURE_DELTA_8BIT
            && different_from_previous >= ABA_CHAIN_MIN_DIFFERENT_DELTA_8BIT
        {
            chain.push(candidate);
        } else {
            break;
        }
    }

    if chain.len() - 1 < ABA_CHAIN_MIN_SEGMENTS {
        return None;
    }

    let suppress = chain[1..chain.len() - 1]
        .iter()
        .copied()
        .filter(|frame| {
            keyframes.contains(frame)
                && scores
                    .get(frame)
                    .copied()
                    .is_some_and(aba_chain_suppressible_cut)
        })
        .collect::<Vec<_>>();
    (!suppress.is_empty()).then_some(AbaChainCandidate {
        boundaries: chain,
        suppress,
    })
}

fn aba_chain_hidden_boundary(
    frame: usize,
    score: ScenecutResult,
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
) -> bool {
    if frame == 0 || keyframes.contains(&frame) {
        return false;
    }
    if !matches!(
        score.decision,
        ScenecutDecision::NoCut
            | ScenecutDecision::SuppressedForwardSimilarity
            | ScenecutDecision::SuppressedTransientSimilarity
    ) {
        return false;
    }
    if score.cost_ratio < ABA_CHAIN_MIN_HIDDEN_COST_RATIO
        || score.cost_ratio > ABA_CHAIN_MAX_HIDDEN_COST_RATIO
        || score.imp_block_ratio < ABA_CHAIN_MIN_HIDDEN_IMP_RATIO
        || score.global_imp_block_ratio < ABA_CHAIN_MIN_HIDDEN_GLOBAL_IMP_RATIO
        || score.avg_luma_8bit > ABA_CHAIN_MAX_HIDDEN_LUMA_8BIT
        || score.static_good_block_ratio > ABA_CHAIN_MAX_HIDDEN_GOOD_BLOCK_RATIO
    {
        return false;
    }
    aba_chain_signature_delta(frame.saturating_sub(1), frame, scores)
        .is_some_and(|delta| delta >= ABA_CHAIN_MIN_HIDDEN_EDGE_DELTA_8BIT)
}

fn aba_chain_suppressible_cut(score: ScenecutResult) -> bool {
    matches!(
        score.decision,
        ScenecutDecision::Cut | ScenecutDecision::CutImportance | ScenecutDecision::CutAbaReturn
    )
}

fn aba_chain_signature_delta(
    left: usize,
    right: usize,
    scores: &BTreeMap<usize, ScenecutResult>,
) -> Option<f64> {
    Some(frame_signature_delta_8bit(
        scores.get(&left)?.frame_luma_signature?,
        scores.get(&right)?.frame_luma_signature?,
    ))
}

const FORWARD_SIMILARITY_RECOVERY_MIN_MATCH_DELTA_8BIT: f64 = 8.0;
const FORWARD_SIMILARITY_RECOVERY_MIN_COST_RATIO: f64 = 1.2;
const FORWARD_SIMILARITY_RECOVERY_MIN_IMP_RATIO: f64 = 5.0;
const FORWARD_SIMILARITY_RECOVERY_MIN_GLOBAL_IMP_RATIO: f64 = 6.0;
const FORWARD_SIMILARITY_RECOVERY_MIN_EDGE_DELTA_8BIT: f64 = 20.0;
const FORWARD_SIMILARITY_RECOVERY_MAX_FRAMES: usize = 3;

fn apply_forward_similarity_recovery_postprocess(
    opts: DetectionOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let min_distance = opts.min_scenecut_distance.unwrap_or(0);
    let candidates = scores
        .iter()
        .filter_map(|(&frame, &score)| {
            forward_similarity_recovery_candidate(frame, score, scores, min_distance)
        })
        .collect::<Vec<_>>();

    for mut frames in candidates {
        frames.sort_unstable();
        frames.dedup();
        if frames.len() > FORWARD_SIMILARITY_RECOVERY_MAX_FRAMES
            || !forward_similarity_recovery_distance_ok(&frames, keyframes, min_distance)
        {
            continue;
        }
        for frame in frames {
            keyframes.insert(frame);
            if let Some(score) = scores.get_mut(&frame) {
                score.decision = ScenecutDecision::CutForwardSimilarityRecovery;
            }
        }
    }
}

fn forward_similarity_recovery_candidate(
    frame: usize,
    score: ScenecutResult,
    scores: &BTreeMap<usize, ScenecutResult>,
    min_distance: usize,
) -> Option<Vec<usize>> {
    if score.decision != ScenecutDecision::SuppressedForwardSimilarity
        || score.forward_similarity_score? < FORWARD_SIMILARITY_RECOVERY_MIN_MATCH_DELTA_8BIT
        || !forward_similarity_recovery_score(frame, score, scores)
    {
        return None;
    }
    let return_frame = score.forward_return_frame?;
    let search_start = frame.saturating_add(min_distance.max(1));
    let search_end = return_frame.saturating_sub(min_distance.max(1));
    if search_start > search_end {
        return None;
    }

    let inside = (search_start..=search_end)
        .filter(|&candidate| {
            scores
                .get(&candidate)
                .copied()
                .is_some_and(|candidate_score| {
                    candidate_score.decision == ScenecutDecision::SuppressedForwardSimilarity
                        && forward_similarity_recovery_score(candidate, candidate_score, scores)
                })
        })
        .collect::<Vec<_>>();
    if inside.is_empty() {
        return None;
    }

    let mut frames = Vec::with_capacity(inside.len() + 1);
    frames.push(frame);
    frames.extend(inside);
    Some(frames)
}

fn forward_similarity_recovery_score(
    frame: usize,
    score: ScenecutResult,
    scores: &BTreeMap<usize, ScenecutResult>,
) -> bool {
    score.cost_ratio >= FORWARD_SIMILARITY_RECOVERY_MIN_COST_RATIO
        && score.imp_block_ratio >= FORWARD_SIMILARITY_RECOVERY_MIN_IMP_RATIO
        && score.global_imp_block_ratio >= FORWARD_SIMILARITY_RECOVERY_MIN_GLOBAL_IMP_RATIO
        && frame_signature_delta_for_scores(frame.saturating_sub(1), frame, scores)
            .is_some_and(|delta| delta >= FORWARD_SIMILARITY_RECOVERY_MIN_EDGE_DELTA_8BIT)
}

fn forward_similarity_recovery_distance_ok(
    frames: &[usize],
    keyframes: &BTreeSet<usize>,
    min_distance: usize,
) -> bool {
    let mut trial = keyframes.clone();
    for &frame in frames {
        if !scene_distance_ok(frame, &trial, min_distance) {
            return false;
        }
        trial.insert(frame);
    }
    true
}

#[derive(Clone, Copy)]
struct BoundaryShiftCandidate {
    from: usize,
    to: usize,
}

const TEXT_BOUNDARY_SHIFT_MIN_FORWARD_COST_RATIO: f64 = 1.0;
const TEXT_BOUNDARY_SHIFT_MIN_FORWARD_IMP_RATIO: f64 = 8.0;
const TEXT_BOUNDARY_SHIFT_MIN_FORWARD_GLOBAL_IMP_RATIO: f64 = 8.0;
const TEXT_BOUNDARY_SHIFT_MAX_FORWARD_LUMA_8BIT: f64 = 40.0;
const TEXT_BOUNDARY_SHIFT_MIN_FORWARD_BAD_BLOCK_RATIO: f64 = 0.75;
const TEXT_BOUNDARY_SHIFT_MAX_FORWARD_GOOD_BLOCK_RATIO: f64 = 0.02;
const TEXT_BOUNDARY_SHIFT_MIN_BACKWARD_IMP_RATIO: f64 = 4.5;
const TEXT_BOUNDARY_SHIFT_MIN_BACKWARD_GLOBAL_IMP_RATIO: f64 = 8.0;
const TEXT_BOUNDARY_SHIFT_MAX_BACKWARD_LUMA_8BIT: f64 = 35.0;
const TEXT_BOUNDARY_SHIFT_MAX_NEXT_COST_RATIO: f64 = 1.0;
const TEXT_BOUNDARY_SHIFT_MIN_NEXT_LUMA_8BIT: f64 = 45.0;
const TEXT_BOUNDARY_SHIFT_MIN_GLOBAL_IMP_MARGIN: f64 = 1.0;

fn apply_text_boundary_shift_postprocess(
    opts: DetectionOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let min_distance = opts.min_scenecut_distance.unwrap_or(0);
    if min_distance == 0 {
        return;
    }

    let candidates = scores
        .iter()
        .filter_map(|(&frame, &score)| {
            text_boundary_shift_candidate(frame, score, keyframes, scores, min_distance)
        })
        .collect::<Vec<_>>();

    for candidate in candidates {
        if !keyframes.contains(&candidate.from) || keyframes.contains(&candidate.to) {
            continue;
        }
        keyframes.remove(&candidate.from);
        if !scene_distance_ok(candidate.to, keyframes, min_distance) {
            keyframes.insert(candidate.from);
            continue;
        }
        keyframes.insert(candidate.to);
        if let Some(score) = scores.get_mut(&candidate.from) {
            score.decision = ScenecutDecision::SuppressedShiftedBoundary;
        }
        if let Some(score) = scores.get_mut(&candidate.to) {
            score.decision = ScenecutDecision::CutShiftedBoundary;
        }
    }
}

fn text_boundary_shift_candidate(
    frame: usize,
    score: ScenecutResult,
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
    min_distance: usize,
) -> Option<BoundaryShiftCandidate> {
    if keyframes.contains(&frame) {
        return None;
    }
    let previous_cut = keyframes.range(..frame).next_back().copied()?;
    let next_cut = keyframes.range((frame + 1)..).next().copied()?;
    if previous_cut == 0 && frame - previous_cut < min_distance {
        return None;
    }

    if previous_cut != 0
        && frame - previous_cut < min_distance
        && next_cut - frame >= min_distance
        && text_boundary_forward_shift_score(score)
    {
        return Some(BoundaryShiftCandidate {
            from: previous_cut,
            to: frame,
        });
    }

    let next_score = scores.get(&next_cut).copied()?;
    if frame - previous_cut >= min_distance
        && next_cut - frame < min_distance
        && text_boundary_backward_shift_score(score, next_score)
    {
        return Some(BoundaryShiftCandidate {
            from: next_cut,
            to: frame,
        });
    }

    None
}

fn text_boundary_forward_shift_score(score: ScenecutResult) -> bool {
    score.decision == ScenecutDecision::SuppressedMinDistance
        && score.cost_ratio >= TEXT_BOUNDARY_SHIFT_MIN_FORWARD_COST_RATIO
        && score.imp_block_ratio >= TEXT_BOUNDARY_SHIFT_MIN_FORWARD_IMP_RATIO
        && score.global_imp_block_ratio >= TEXT_BOUNDARY_SHIFT_MIN_FORWARD_GLOBAL_IMP_RATIO
        && score.avg_luma_8bit <= TEXT_BOUNDARY_SHIFT_MAX_FORWARD_LUMA_8BIT
        && score.static_bad_block_ratio >= TEXT_BOUNDARY_SHIFT_MIN_FORWARD_BAD_BLOCK_RATIO
        && score.static_good_block_ratio <= TEXT_BOUNDARY_SHIFT_MAX_FORWARD_GOOD_BLOCK_RATIO
}

fn text_boundary_backward_shift_score(score: ScenecutResult, next_score: ScenecutResult) -> bool {
    matches!(
        score.decision,
        ScenecutDecision::NoCut
            | ScenecutDecision::SuppressedImportance
            | ScenecutDecision::SuppressedMinDistance
    ) && score.imp_block_ratio >= TEXT_BOUNDARY_SHIFT_MIN_BACKWARD_IMP_RATIO
        && score.global_imp_block_ratio >= TEXT_BOUNDARY_SHIFT_MIN_BACKWARD_GLOBAL_IMP_RATIO
        && score.avg_luma_8bit <= TEXT_BOUNDARY_SHIFT_MAX_BACKWARD_LUMA_8BIT
        && next_score.cost_ratio <= TEXT_BOUNDARY_SHIFT_MAX_NEXT_COST_RATIO
        && next_score.avg_luma_8bit >= TEXT_BOUNDARY_SHIFT_MIN_NEXT_LUMA_8BIT
        && score.global_imp_block_ratio - next_score.global_imp_block_ratio
            >= TEXT_BOUNDARY_SHIFT_MIN_GLOBAL_IMP_MARGIN
}

const STATIC_CREDITS_MIN_RUN_CUTS: usize = 5;
const STATIC_CREDITS_MAX_GAP: usize = 210;
const STATIC_CREDITS_MAX_LUMA_8BIT: f64 = 45.0;
const STATIC_CREDITS_MIN_IMP_RATIO: f64 = 6.0;
const STATIC_CREDITS_MIN_GLOBAL_IMP_RATIO: f64 = 8.0;
const STATIC_CREDITS_MIN_GOOD_BLOCK_RATIO: f64 = 0.35;
const STATIC_CREDITS_MAX_BAD_BLOCK_RATIO: f64 = 0.65;

fn apply_static_credits_postprocess(
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    let cuts = keyframes
        .iter()
        .copied()
        .filter(|&frame| frame != 0)
        .collect::<Vec<_>>();
    let mut run = Vec::new();

    for frame in cuts {
        let static_credits_cut = scores.get(&frame).copied().is_some_and(static_credits_cut);
        let extends_run = run
            .last()
            .is_none_or(|previous| frame - previous <= STATIC_CREDITS_MAX_GAP);
        if static_credits_cut && extends_run {
            run.push(frame);
        } else {
            suppress_static_credits_run(&run, keyframes, scores);
            run.clear();
            if static_credits_cut {
                run.push(frame);
            }
        }
    }

    suppress_static_credits_run(&run, keyframes, scores);
}

fn suppress_static_credits_run(
    run: &[usize],
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    if run.len() < STATIC_CREDITS_MIN_RUN_CUTS {
        return;
    }

    for &frame in &run[1..] {
        keyframes.remove(&frame);
        if let Some(score) = scores.get_mut(&frame) {
            score.decision = ScenecutDecision::SuppressedStaticCredits;
        }
    }
}

fn static_credits_cut(score: ScenecutResult) -> bool {
    score.decision == ScenecutDecision::Cut
        && score.avg_luma_8bit <= STATIC_CREDITS_MAX_LUMA_8BIT
        && score.imp_block_ratio >= STATIC_CREDITS_MIN_IMP_RATIO
        && score.global_imp_block_ratio >= STATIC_CREDITS_MIN_GLOBAL_IMP_RATIO
        && score.static_good_block_ratio >= STATIC_CREDITS_MIN_GOOD_BLOCK_RATIO
        && score.static_bad_block_ratio <= STATIC_CREDITS_MAX_BAD_BLOCK_RATIO
}

/// Specifies the scene detection algorithm to use
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SceneDetectionSpeed {
    /// Fastest scene detection using pixel-wise comparison
    Fast,
    /// Scene detection using frame costs and motion vectors
    Standard,
    /// Higher quality cost-based detection with additional dark-scene and
    /// forward-similarity heuristics.
    High,
    /// Do not perform scenecut detection, only place keyframes at fixed
    /// intervals
    None,
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, Read},
        sync::Mutex,
    };

    use av_decoders::Y4mDecoder;

    use super::*;

    const TEST_FILE: &str = "./test_files/tt_sif.y4m";

    fn decoder() -> Decoder {
        let file = File::open(TEST_FILE).expect("test y4m should exist");
        let reader = BufReader::new(file);
        Decoder::from_decoder_impl(av_decoders::DecoderImpl::Y4m(
            Y4mDecoder::new(Box::new(reader) as Box<dyn Read>).expect("y4m should decode"),
        ))
        .expect("decoder should initialize")
    }

    fn standard_options() -> DetectionOptions {
        DetectionOptions {
            min_scenecut_distance: Some(24),
            max_scenecut_distance: None,
            ..DetectionOptions::default()
        }
    }

    fn score(
        decision: ScenecutDecision,
        cost_ratio: f64,
        imp_block_ratio: f64,
        global_imp_block_ratio: f64,
        avg_luma_8bit: f64,
        transient_similarity_score: Option<f64>,
    ) -> ScenecutResult {
        let mut score = ScenecutResult::new(0.0, 0.0, 1.0, 1.0, avg_luma_8bit);
        score.decision = decision;
        score.cost_ratio = cost_ratio;
        score.imp_block_ratio = imp_block_ratio;
        score.global_imp_block_ratio = global_imp_block_ratio;
        score.transient_similarity_score = transient_similarity_score;
        score
    }

    fn static_credit_score() -> ScenecutResult {
        let mut score = score(ScenecutDecision::Cut, 2.0, 9.0, 10.0, 28.0, None);
        score.static_good_block_ratio = 0.6;
        score.static_bad_block_ratio = 0.35;
        score
    }

    fn dark_occlusion_score(with_forward_candidate: bool) -> ScenecutResult {
        let mut score = score(
            ScenecutDecision::CutImportance,
            0.18,
            3.7,
            4.7,
            24.0,
            Some(3.5),
        );
        score.static_bad_block_ratio = 0.75;
        score.static_good_block_ratio = 0.0;
        if with_forward_candidate {
            score.forward_similarity_candidates[0] = Some(ForwardSimilarityCandidate {
                frame: 110,
                offset: 10,
                delta: 4.1,
                threshold: 6.0,
                candidate_frame: None,
                candidate_offset: None,
                decision: ForwardSimilarityCandidateDecision::MissingReturnCandidate,
            });
        }
        score
    }

    fn dark_scene_peak_score_for_test(cost_ratio: f64) -> ScenecutResult {
        let mut score = score(ScenecutDecision::NoCut, cost_ratio, 1.8, 1.9, 43.0, None);
        score.static_bad_block_ratio = 0.72;
        score.static_good_block_ratio = 0.001;
        score
    }

    fn sparse_scene_peak_score_for_test(cost_ratio: f64) -> ScenecutResult {
        let mut score = score(ScenecutDecision::NoCut, cost_ratio, 3.2, 3.8, 72.0, None);
        score.static_bad_block_ratio = 0.62;
        score.static_good_block_ratio = 0.02;
        score
    }

    fn refined_sparse_peak_score_for_test(
        decision: ScenecutDecision,
        signature_value: u8,
        cost_ratio: f64,
    ) -> ScenecutResult {
        let mut score = score_with_signature(decision, signature_value);
        score.cost_ratio = cost_ratio;
        score.imp_block_ratio = 4.2;
        score.global_imp_block_ratio = 4.8;
        score.avg_luma_8bit = 70.0;
        score.static_bad_block_ratio = 0.56;
        score.static_good_block_ratio = 0.03;
        score
    }

    fn forward_recovery_score_for_test(
        signature_value: u8,
        forward_score: Option<f64>,
        return_frame: Option<usize>,
    ) -> ScenecutResult {
        let mut score = score_with_signature(
            ScenecutDecision::SuppressedForwardSimilarity,
            signature_value,
        );
        score.cost_ratio = 1.5;
        score.imp_block_ratio = 5.5;
        score.global_imp_block_ratio = 6.5;
        score.forward_similarity_score = forward_score;
        score.forward_return_frame = return_frame;
        score
    }

    fn score_with_signature(decision: ScenecutDecision, signature_value: u8) -> ScenecutResult {
        let mut score = score(decision, 0.7, 4.0, 5.0, 50.0, None);
        score.static_good_block_ratio = 0.05;
        score.frame_luma_signature = Some([signature_value; analyze::FRAME_LUMA_SIGNATURE_CELLS]);
        score
    }

    #[test]
    fn text_card_postprocess_suppresses_only_weak_runs() {
        let mut keyframes = BTreeSet::from([0, 100, 220, 244, 340, 520, 650]);
        let mut scores = BTreeMap::from([
            (
                100,
                score(ScenecutDecision::Cut, 1.1, 9.0, 10.0, 18.0, None),
            ),
            (
                220,
                score(
                    ScenecutDecision::CutImportance,
                    0.14,
                    2.9,
                    4.5,
                    28.0,
                    Some(2.1),
                ),
            ),
            (
                244,
                score(
                    ScenecutDecision::CutImportance,
                    0.17,
                    2.8,
                    3.9,
                    31.0,
                    Some(2.0),
                ),
            ),
            (
                340,
                score(
                    ScenecutDecision::CutImportance,
                    0.37,
                    2.6,
                    4.5,
                    29.0,
                    Some(1.7),
                ),
            ),
            (520, score(ScenecutDecision::Cut, 1.2, 6.0, 7.0, 50.0, None)),
            (
                650,
                score(
                    ScenecutDecision::CutImportance,
                    0.18,
                    3.7,
                    4.7,
                    24.0,
                    Some(3.5),
                ),
            ),
        ]);

        apply_text_card_cluster_postprocess(&mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 100, 520, 650]));
        for frame in [220, 244, 340] {
            assert_eq!(
                scores.get(&frame).expect("score should exist").decision,
                ScenecutDecision::SuppressedTextCardCluster
            );
        }
        assert_eq!(
            scores.get(&650).expect("score should exist").decision,
            ScenecutDecision::CutImportance
        );
    }

    #[test]
    fn fast_motion_postprocess_suppresses_micro_split_pair() {
        let mut keyframes = BTreeSet::from([0, 1000, 2259, 2276, 2326, 2600]);
        let mut scores = BTreeMap::from([
            (
                1000,
                score(ScenecutDecision::Cut, 1.5, 5.0, 5.0, 55.0, None),
            ),
            (
                2259,
                score(
                    ScenecutDecision::CutImportance,
                    0.19,
                    3.1,
                    4.5,
                    57.0,
                    Some(6.8),
                ),
            ),
            (
                2276,
                score(
                    ScenecutDecision::CutImportance,
                    0.28,
                    3.5,
                    4.9,
                    63.0,
                    Some(6.6),
                ),
            ),
            (
                2326,
                score(ScenecutDecision::Cut, 3.1, 2.9, 3.0, 64.0, None),
            ),
            (
                2600,
                score(
                    ScenecutDecision::CutImportance,
                    0.20,
                    3.2,
                    4.3,
                    58.0,
                    Some(6.4),
                ),
            ),
        ]);

        apply_fast_motion_micro_split_postprocess(&mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 1000, 2326, 2600]));
        for frame in [2259, 2276] {
            assert_eq!(
                scores.get(&frame).expect("score should exist").decision,
                ScenecutDecision::SuppressedFastMotion
            );
        }
    }

    #[test]
    fn forward_similarity_postprocess_uses_already_suppressed_starts() {
        let mut keyframes = BTreeSet::from([0, 150, 300]);
        let mut start = score(
            ScenecutDecision::SuppressedForwardSimilarity,
            0.7,
            4.2,
            5.0,
            50.0,
            None,
        );
        start.forward_similarity_candidates[0] = Some(ForwardSimilarityCandidate {
            frame: 220,
            offset: 101,
            delta: 2.0,
            threshold: 6.0,
            candidate_frame: Some(150),
            candidate_offset: Some(31),
            decision: ForwardSimilarityCandidateDecision::Accepted,
        });
        let mut scores = BTreeMap::from([
            (120, start),
            (150, score(ScenecutDecision::Cut, 0.8, 4.2, 5.0, 50.0, None)),
            (
                220,
                score(ScenecutDecision::NoCut, 0.1, 1.0, 1.0, 50.0, None),
            ),
            (300, score(ScenecutDecision::Cut, 1.2, 4.2, 5.0, 50.0, None)),
        ]);
        let options = ForwardSimilarityOptions {
            enabled: true,
            frames: 120,
            min_offset: 4,
            threshold_8bit: 6.0,
            require_return_candidate: true,
            suppress_inside: true,
            ..ForwardSimilarityOptions::default()
        };

        apply_forward_similarity_postprocess(options, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 300]));
        assert_eq!(
            scores.get(&150).expect("score should exist").decision,
            ScenecutDecision::SuppressedForwardSimilarity
        );
        assert_eq!(
            scores
                .get(&150)
                .expect("score should exist")
                .forward_return_frame,
            Some(220)
        );
    }

    #[test]
    fn forward_similarity_recovery_restores_relaxed_false_positive_pair() {
        let mut keyframes = BTreeSet::from([0, 220]);
        let mut scores = BTreeMap::from([
            (79, score_with_signature(ScenecutDecision::NoCut, 10)),
            (
                80,
                forward_recovery_score_for_test(40, Some(8.1), Some(170)),
            ),
            (119, score_with_signature(ScenecutDecision::NoCut, 40)),
            (120, forward_recovery_score_for_test(80, None, Some(170))),
            (220, score_with_signature(ScenecutDecision::Cut, 120)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_forward_similarity_recovery_postprocess(opts, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 80, 120, 220]));
        for frame in [80, 120] {
            assert_eq!(
                scores.get(&frame).expect("score should exist").decision,
                ScenecutDecision::CutForwardSimilarityRecovery
            );
        }
    }

    #[test]
    fn forward_similarity_recovery_keeps_strict_match_suppressed() {
        let mut keyframes = BTreeSet::from([0, 220]);
        let mut scores = BTreeMap::from([
            (79, score_with_signature(ScenecutDecision::NoCut, 10)),
            (
                80,
                forward_recovery_score_for_test(40, Some(5.9), Some(170)),
            ),
            (119, score_with_signature(ScenecutDecision::NoCut, 40)),
            (120, forward_recovery_score_for_test(80, None, Some(170))),
            (220, score_with_signature(ScenecutDecision::Cut, 120)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_forward_similarity_recovery_postprocess(opts, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 220]));
        assert_eq!(
            scores.get(&80).expect("score should exist").decision,
            ScenecutDecision::SuppressedForwardSimilarity
        );
    }

    #[test]
    fn dark_occlusion_postprocess_requires_forward_similarity_evidence() {
        let mut keyframes = BTreeSet::from([0, 100, 200]);
        let mut scores = BTreeMap::from([
            (100, dark_occlusion_score(true)),
            (200, dark_occlusion_score(false)),
        ]);

        apply_dark_occlusion_postprocess(&mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 200]));
        assert_eq!(
            scores.get(&100).expect("score should exist").decision,
            ScenecutDecision::SuppressedDarkOcclusion
        );
        assert_eq!(
            scores.get(&200).expect("score should exist").decision,
            ScenecutDecision::CutImportance
        );
    }

    #[test]
    fn dark_scene_peak_recovery_promotes_sparse_dark_local_peak() {
        let mut keyframes = BTreeSet::from([0, 100, 700]);
        let mut scores = BTreeMap::from([
            (
                100,
                score(ScenecutDecision::CutImportance, 0.6, 3.0, 3.5, 48.0, None),
            ),
            (390, dark_scene_peak_score_for_test(0.33)),
            (391, dark_scene_peak_score_for_test(0.25)),
            (700, score(ScenecutDecision::Cut, 3.0, 3.0, 3.5, 80.0, None)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_dark_scene_peak_recovery_postprocess(opts, &mut keyframes, &mut scores);

        assert!(keyframes.contains(&390));
        assert_eq!(
            scores.get(&390).expect("score should exist").decision,
            ScenecutDecision::CutDarkScenePeak
        );
        assert_eq!(
            scores.get(&391).expect("score should exist").decision,
            ScenecutDecision::NoCut
        );
    }

    #[test]
    fn dark_scene_peak_recovery_rejects_dense_cut_regions() {
        let mut keyframes = BTreeSet::from([0, 100, 160, 220, 280, 340, 400, 460, 520, 580, 640]);
        let mut scores = BTreeMap::new();
        for &frame in &keyframes {
            if frame != 0 {
                scores.insert(
                    frame,
                    score(ScenecutDecision::Cut, 1.0, 3.0, 3.5, 50.0, None),
                );
            }
        }
        scores.insert(370, dark_scene_peak_score_for_test(0.35));
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_dark_scene_peak_recovery_postprocess(opts, &mut keyframes, &mut scores);

        assert!(!keyframes.contains(&370));
        assert_eq!(
            scores.get(&370).expect("score should exist").decision,
            ScenecutDecision::NoCut
        );
    }

    #[test]
    fn sparse_scene_peak_recovery_promotes_local_peak() {
        let mut keyframes = BTreeSet::from([0, 100, 360]);
        let mut scores = BTreeMap::from([
            (100, score(ScenecutDecision::Cut, 1.4, 4.0, 4.5, 70.0, None)),
            (220, sparse_scene_peak_score_for_test(0.82)),
            (221, sparse_scene_peak_score_for_test(0.70)),
            (360, score(ScenecutDecision::Cut, 1.3, 4.0, 4.5, 70.0, None)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_sparse_scene_peak_recovery_postprocess(opts, &mut keyframes, &mut scores);

        assert!(keyframes.contains(&220));
        assert_eq!(
            scores.get(&220).expect("score should exist").decision,
            ScenecutDecision::CutSparseScenePeak
        );
        assert_eq!(
            scores.get(&221).expect("score should exist").decision,
            ScenecutDecision::NoCut
        );
    }

    #[test]
    fn sparse_scene_peak_recovery_rejects_extreme_adjacent_cut() {
        let mut keyframes = BTreeSet::from([0, 100, 360]);
        let mut scores = BTreeMap::from([
            (
                100,
                score(ScenecutDecision::Cut, 20.0, 8.0, 8.0, 80.0, None),
            ),
            (220, sparse_scene_peak_score_for_test(0.82)),
            (360, score(ScenecutDecision::Cut, 1.3, 4.0, 4.5, 70.0, None)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_sparse_scene_peak_recovery_postprocess(opts, &mut keyframes, &mut scores);

        assert!(!keyframes.contains(&220));
        assert_eq!(
            scores.get(&220).expect("score should exist").decision,
            ScenecutDecision::NoCut
        );
    }

    #[test]
    fn refined_sparse_peak_shifts_return_boundary_to_exit_boundary() {
        let mut keyframes = BTreeSet::from([0, 100, 160, 260, 340]);
        let mut scores = BTreeMap::from([
            (
                100,
                refined_sparse_peak_score_for_test(ScenecutDecision::Cut, 70, 1.2),
            ),
            (
                160,
                refined_sparse_peak_score_for_test(ScenecutDecision::CutSparseScenePeak, 10, 0.9),
            ),
            (
                177,
                refined_sparse_peak_score_for_test(ScenecutDecision::NoCut, 10, 0.1),
            ),
            (
                178,
                refined_sparse_peak_score_for_test(ScenecutDecision::NoCut, 40, 0.8),
            ),
            (
                260,
                refined_sparse_peak_score_for_test(ScenecutDecision::Cut, 80, 1.1),
            ),
            (
                340,
                refined_sparse_peak_score_for_test(ScenecutDecision::Cut, 10, 1.1),
            ),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_refined_sparse_peak_postprocess(opts, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 100, 178, 260, 340]));
        assert_eq!(
            scores.get(&160).expect("score should exist").decision,
            ScenecutDecision::SuppressedRefinedSparsePeak
        );
        assert_eq!(
            scores.get(&178).expect("score should exist").decision,
            ScenecutDecision::CutRefinedSparsePeak
        );
    }

    #[test]
    fn refined_sparse_peak_suppresses_short_aba_sparse_cut() {
        let mut keyframes = BTreeSet::from([0, 220, 310, 500]);
        let mut scores = BTreeMap::from([
            (
                149,
                refined_sparse_peak_score_for_test(ScenecutDecision::NoCut, 20, 0.1),
            ),
            (
                150,
                refined_sparse_peak_score_for_test(ScenecutDecision::NoCut, 50, 0.8),
            ),
            (
                220,
                refined_sparse_peak_score_for_test(ScenecutDecision::CutSparseScenePeak, 80, 0.9),
            ),
            (
                244,
                refined_sparse_peak_score_for_test(ScenecutDecision::NoCut, 80, 0.1),
            ),
            (
                245,
                refined_sparse_peak_score_for_test(ScenecutDecision::NoCut, 30, 0.85),
            ),
            (
                310,
                refined_sparse_peak_score_for_test(ScenecutDecision::Cut, 60, 1.2),
            ),
            (
                500,
                refined_sparse_peak_score_for_test(ScenecutDecision::Cut, 90, 1.2),
            ),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_refined_sparse_peak_postprocess(opts, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 150, 310, 500]));
        assert_eq!(
            scores.get(&150).expect("score should exist").decision,
            ScenecutDecision::CutRefinedSparsePeak
        );
        assert_eq!(
            scores.get(&220).expect("score should exist").decision,
            ScenecutDecision::SuppressedRefinedSparsePeak
        );
        assert_eq!(
            scores.get(&245).expect("score should exist").decision,
            ScenecutDecision::NoCut
        );
    }

    #[test]
    fn refined_sparse_peak_rejects_low_cost_motion_peak() {
        let mut keyframes = BTreeSet::from([0, 100, 300]);
        let mut candidate = refined_sparse_peak_score_for_test(ScenecutDecision::NoCut, 80, 0.41);
        candidate.imp_block_ratio = 3.4;
        candidate.global_imp_block_ratio = 4.1;
        let mut scores = BTreeMap::from([
            (
                199,
                refined_sparse_peak_score_for_test(ScenecutDecision::NoCut, 40, 0.1),
            ),
            (200, candidate),
            (
                300,
                refined_sparse_peak_score_for_test(ScenecutDecision::Cut, 120, 1.2),
            ),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_refined_sparse_peak_postprocess(opts, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 100, 300]));
        assert_eq!(
            scores.get(&200).expect("score should exist").decision,
            ScenecutDecision::NoCut
        );
    }

    #[test]
    fn aba_return_recovery_promotes_signature_match() {
        let mut keyframes = BTreeSet::from([0, 100, 300]);
        let mut scores = BTreeMap::from([
            (99, score_with_signature(ScenecutDecision::NoCut, 10)),
            (100, score_with_signature(ScenecutDecision::Cut, 40)),
            (199, score_with_signature(ScenecutDecision::NoCut, 40)),
            (
                200,
                score_with_signature(ScenecutDecision::SuppressedTransientSimilarity, 10),
            ),
            (300, score_with_signature(ScenecutDecision::Cut, 80)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_aba_return_recovery_postprocess(opts, &mut keyframes, &mut scores);

        assert!(keyframes.contains(&200));
        assert_eq!(
            scores.get(&200).expect("score should exist").decision,
            ScenecutDecision::CutAbaReturn
        );
    }

    #[test]
    fn aba_return_recovery_keeps_one_candidate_per_interval() {
        let mut keyframes = BTreeSet::from([0, 100, 400]);
        let mut scores = BTreeMap::from([
            (99, score_with_signature(ScenecutDecision::NoCut, 10)),
            (100, score_with_signature(ScenecutDecision::Cut, 40)),
            (199, score_with_signature(ScenecutDecision::NoCut, 40)),
            (200, score_with_signature(ScenecutDecision::NoCut, 11)),
            (259, score_with_signature(ScenecutDecision::NoCut, 40)),
            (260, score_with_signature(ScenecutDecision::NoCut, 13)),
            (400, score_with_signature(ScenecutDecision::Cut, 80)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_aba_return_recovery_postprocess(opts, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 100, 200, 400]));
        assert_eq!(
            scores.get(&200).expect("score should exist").decision,
            ScenecutDecision::CutAbaReturn
        );
        assert_eq!(
            scores.get(&260).expect("score should exist").decision,
            ScenecutDecision::NoCut
        );
    }

    #[test]
    fn aba_chain_compaction_splits_after_long_segment() {
        let mut keyframes = BTreeSet::from([0, 100, 190, 285, 410, 470, 560]);
        let mut scores = BTreeMap::from([
            (100, score_with_signature(ScenecutDecision::Cut, 10)),
            (129, score_with_signature(ScenecutDecision::NoCut, 10)),
            (130, score_with_signature(ScenecutDecision::NoCut, 40)),
            (159, score_with_signature(ScenecutDecision::NoCut, 40)),
            (160, score_with_signature(ScenecutDecision::NoCut, 10)),
            (
                190,
                score_with_signature(ScenecutDecision::CutImportance, 40),
            ),
            (285, score_with_signature(ScenecutDecision::Cut, 10)),
            (
                410,
                score_with_signature(ScenecutDecision::CutAbaReturn, 40),
            ),
            (429, score_with_signature(ScenecutDecision::NoCut, 40)),
            (430, score_with_signature(ScenecutDecision::NoCut, 10)),
            (449, score_with_signature(ScenecutDecision::NoCut, 10)),
            (450, score_with_signature(ScenecutDecision::NoCut, 40)),
            (470, score_with_signature(ScenecutDecision::Cut, 10)),
            (560, score_with_signature(ScenecutDecision::Cut, 80)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.tuning.forward_similarity.frames = 80;

        apply_aba_chain_compaction_postprocess(opts, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 100, 285, 470, 560]));
        for frame in [190, 410] {
            assert_eq!(
                scores.get(&frame).expect("score should exist").decision,
                ScenecutDecision::SuppressedAbaChain
            );
        }
        assert_eq!(
            scores.get(&285).expect("score should exist").decision,
            ScenecutDecision::Cut
        );
        assert_eq!(
            scores.get(&470).expect("score should exist").decision,
            ScenecutDecision::Cut
        );
    }

    #[test]
    fn text_boundary_shift_moves_late_boundary_back() {
        let mut keyframes = BTreeSet::from([0, 100, 316, 380]);
        let mut candidate = score(ScenecutDecision::NoCut, 0.0, 5.1, 8.9, 29.0, None);
        candidate.static_bad_block_ratio = 0.35;
        candidate.static_good_block_ratio = 0.03;
        let mut late_cut = score(
            ScenecutDecision::CutImportance,
            0.65,
            6.3,
            7.2,
            50.0,
            Some(15.0),
        );
        late_cut.static_bad_block_ratio = 0.65;
        late_cut.static_good_block_ratio = 0.02;
        let mut scores = BTreeMap::from([
            (100, score(ScenecutDecision::Cut, 1.4, 4.0, 4.5, 70.0, None)),
            (300, candidate),
            (316, late_cut),
            (380, score(ScenecutDecision::Cut, 1.4, 4.0, 4.5, 70.0, None)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_text_boundary_shift_postprocess(opts, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 100, 300, 380]));
        assert_eq!(
            scores.get(&300).expect("score should exist").decision,
            ScenecutDecision::CutShiftedBoundary
        );
        assert_eq!(
            scores.get(&316).expect("score should exist").decision,
            ScenecutDecision::SuppressedShiftedBoundary
        );
    }

    #[test]
    fn text_boundary_shift_moves_early_boundary_forward() {
        let mut keyframes = BTreeSet::from([0, 100, 115, 240]);
        let mut candidate = score(
            ScenecutDecision::SuppressedMinDistance,
            1.3,
            13.8,
            14.2,
            28.0,
            None,
        );
        candidate.static_bad_block_ratio = 0.88;
        candidate.static_good_block_ratio = 0.001;
        let mut scores = BTreeMap::from([
            (100, score(ScenecutDecision::Cut, 1.4, 4.0, 4.5, 70.0, None)),
            (115, score(ScenecutDecision::Cut, 2.0, 4.5, 4.8, 83.0, None)),
            (130, candidate),
            (240, score(ScenecutDecision::Cut, 1.4, 4.0, 4.5, 70.0, None)),
        ]);
        let mut opts = DetectionOptions::high_quality();
        opts.min_scenecut_distance = Some(17);

        apply_text_boundary_shift_postprocess(opts, &mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 100, 130, 240]));
        assert_eq!(
            scores.get(&130).expect("score should exist").decision,
            ScenecutDecision::CutShiftedBoundary
        );
        assert_eq!(
            scores.get(&115).expect("score should exist").decision,
            ScenecutDecision::SuppressedShiftedBoundary
        );
    }

    #[test]
    fn static_credits_postprocess_keeps_first_cut_in_long_run() {
        let mut keyframes = BTreeSet::from([0, 100, 197, 294, 391, 488, 700, 797, 900]);
        let mut scores = BTreeMap::from([
            (100, static_credit_score()),
            (197, static_credit_score()),
            (294, static_credit_score()),
            (391, static_credit_score()),
            (488, static_credit_score()),
            (700, static_credit_score()),
            (797, static_credit_score()),
            (900, score(ScenecutDecision::Cut, 2.0, 8.0, 8.5, 70.0, None)),
        ]);

        apply_static_credits_postprocess(&mut keyframes, &mut scores);

        assert_eq!(keyframes, BTreeSet::from([0, 100, 700, 797, 900]));
        for frame in [197, 294, 391, 488] {
            assert_eq!(
                scores.get(&frame).expect("score should exist").decision,
                ScenecutDecision::SuppressedStaticCredits
            );
        }
        assert_eq!(
            scores.get(&797).expect("score should exist").decision,
            ScenecutDecision::Cut
        );
    }

    #[test]
    fn parallel_standard_matches_sequential() {
        let mut sequential_decoder = decoder();
        let mut parallel_decoder = decoder();
        let options = standard_options();
        let sequential = detect_scene_changes::<u8>(&mut sequential_decoder, options, None, None)
            .expect("sequential detection should work");
        let frame_count = sequential.frame_count;
        let parallel = detect_scene_changes_parallel::<u8>(
            &mut parallel_decoder,
            options,
            Some(frame_count),
            ParallelDetectionOptions { workers: 3 },
            None,
        )
        .expect("parallel detection should work");

        assert_eq!(parallel.scene_changes, sequential.scene_changes);
        assert_eq!(parallel.scores, sequential.scores);
        assert_eq!(parallel.frame_count, sequential.frame_count);
    }

    #[test]
    fn parallel_high_matches_sequential() {
        let mut sequential_decoder = decoder();
        let mut parallel_decoder = decoder();
        let mut options = DetectionOptions::high_quality();
        options.min_scenecut_distance = Some(24);
        let sequential = detect_scene_changes::<u8>(&mut sequential_decoder, options, None, None)
            .expect("sequential detection should work");
        let frame_count = sequential.frame_count;
        let parallel = detect_scene_changes_parallel::<u8>(
            &mut parallel_decoder,
            options,
            Some(frame_count),
            ParallelDetectionOptions { workers: 3 },
            None,
        )
        .expect("parallel detection should work");

        assert_eq!(parallel.scene_changes, sequential.scene_changes);
        assert_eq!(parallel.scores, sequential.scores);
        assert_eq!(parallel.frame_count, sequential.frame_count);
    }

    #[test]
    fn parallel_reports_incremental_progress() {
        let mut sequential_decoder = decoder();
        let mut parallel_decoder = decoder();
        let options = standard_options();
        let sequential = detect_scene_changes::<u8>(&mut sequential_decoder, options, None, None)
            .expect("sequential detection should work");
        let updates = Mutex::new(Vec::new());

        let parallel = detect_scene_changes_parallel::<u8>(
            &mut parallel_decoder,
            options,
            Some(sequential.frame_count),
            ParallelDetectionOptions { workers: 3 },
            Some(&|frames, _keyframes| {
                updates
                    .lock()
                    .expect("updates lock should work")
                    .push(frames);
            }),
        )
        .expect("parallel detection should work");

        let updates = updates.into_inner().expect("updates lock should work");
        assert_eq!(parallel.frame_count, sequential.frame_count);
        assert_eq!(updates.last().copied(), Some(sequential.frame_count));
        assert!(
            updates
                .iter()
                .any(|&frames| frames > 0 && frames < sequential.frame_count)
        );
    }

    #[test]
    fn parallel_fast_matches_sequential() {
        // Fast + default tuning (forward/transient disabled) makes the parallel
        // reader drop chroma from the store. The fixture is 240p so no downscale
        // happens, but this still guards that the luma-only store reproduces the
        // serial Fast result exactly (P2 chroma-drop parity).
        let mut sequential_decoder = decoder();
        let mut parallel_decoder = decoder();
        let mut options = standard_options();
        options.analysis_speed = SceneDetectionSpeed::Fast;
        let sequential = detect_scene_changes::<u8>(&mut sequential_decoder, options, None, None)
            .expect("sequential detection should work");
        let frame_count = sequential.frame_count;
        let parallel = detect_scene_changes_parallel::<u8>(
            &mut parallel_decoder,
            options,
            Some(frame_count),
            ParallelDetectionOptions { workers: 3 },
            None,
        )
        .expect("parallel detection should work");

        assert_eq!(parallel.scene_changes, sequential.scene_changes);
        assert_eq!(parallel.scores, sequential.scores);
        assert_eq!(parallel.frame_count, sequential.frame_count);
    }

    #[test]
    fn resolve_parallel_frame_count_clamps_sentinel_and_falls_back() {
        // `usize::MAX` is the serial "no limit" sentinel: with a known length it
        // collapses to that length, without one it signals serial fallback (None).
        assert_eq!(
            resolve_parallel_frame_count(Some(usize::MAX), Some(100)),
            Some(100)
        );
        assert_eq!(resolve_parallel_frame_count(Some(usize::MAX), None), None);
        // A real limit is clamped by a smaller known length, else kept as-is.
        assert_eq!(resolve_parallel_frame_count(Some(500), Some(100)), Some(100));
        assert_eq!(resolve_parallel_frame_count(Some(40), Some(100)), Some(40));
        assert_eq!(resolve_parallel_frame_count(Some(40), None), Some(40));
        // No limit: use the known length, or fall back when neither is known.
        assert_eq!(resolve_parallel_frame_count(None, Some(100)), Some(100));
        assert_eq!(resolve_parallel_frame_count(None, None), None);
    }

    #[test]
    fn parallel_chunk_starts_no_overflow_on_sentinel_count() {
        // `idx * frame_count` overflowed (debug panic / release garbage) for a
        // near-`usize::MAX` count before the u128 widening (C7).
        for &workers in &[2usize, 3, 7, 8, 64] {
            let starts = parallel_chunk_starts(usize::MAX, workers);
            assert_eq!(starts.first().copied(), Some(0));
            assert!(starts.windows(2).all(|w| w[0] < w[1]));
            assert!(starts.len() <= workers + 1);
        }
    }

    #[test]
    fn parallel_chunk_starts_matches_naive_formula_for_normal_inputs() {
        // The u128 widening must be byte-identical to the original
        // `idx * frame_count / workers` for every non-overflowing input.
        for frame_count in 0usize..512 {
            for workers in 0usize..9 {
                let expected: Vec<usize> = if workers <= 1 || frame_count == 0 {
                    vec![0]
                } else {
                    let mut starts: Vec<usize> =
                        (0..workers).map(|idx| idx * frame_count / workers).collect();
                    starts.dedup();
                    if starts.first().copied() != Some(0) {
                        starts.insert(0, 0);
                    }
                    starts
                };
                assert_eq!(parallel_chunk_starts(frame_count, workers), expected);
            }
        }
    }

    #[test]
    fn reconcile_caps_progress_bitmap_for_sentinel_limit() {
        // `vec![false; frame_limit]` aborted on a `usize::MAX` limit before the
        // cap (C7). With no messages the channel closes immediately and reconcile
        // returns the "ended before complete" error — the point is that it must
        // allocate (capped) and return, not abort/OOM.
        let (tx, rx) = channel();
        drop(tx);
        let chunk_starts = vec![0usize, 1];
        let stop_after = vec![
            Arc::new(AtomicUsize::new(usize::MAX)),
            Arc::new(AtomicUsize::new(usize::MAX)),
        ];
        let result = reconcile_parallel_workers(
            rx,
            standard_options(),
            &chunk_starts,
            usize::MAX,
            &stop_after,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parallel_sentinel_frame_limit_matches_sequential() {
        // `Some(usize::MAX)` is a valid "no limit" sentinel; it must not overflow
        // or over-allocate, and must reproduce the serial result (C7). On the y4m
        // fixture (`total_frames == None`) this resolves to the serial fallback.
        let mut sequential_decoder = decoder();
        let mut parallel_decoder = decoder();
        let options = standard_options();
        let sequential = detect_scene_changes::<u8>(&mut sequential_decoder, options, None, None)
            .expect("sequential detection should work");
        let parallel = detect_scene_changes_parallel::<u8>(
            &mut parallel_decoder,
            options,
            Some(usize::MAX),
            ParallelDetectionOptions { workers: 3 },
            None,
        )
        .expect("parallel detection should work");

        assert_eq!(parallel.scene_changes, sequential.scene_changes);
        assert_eq!(parallel.scores, sequential.scores);
        assert_eq!(parallel.frame_count, sequential.frame_count);
    }

    #[test]
    fn streamed_reader_buffer_budget_scales_with_active_workers() {
        // P1: the streamed resident-range budget is one per-worker window times
        // the number of active workers (>=1), saturating. Scaling by the active
        // count is what lets the forward-only reader decode far enough ahead to
        // feed later chunk workers instead of stalling one window past worker 0.
        assert_eq!(parallel_streamed_reader_buffer_frames(123, 0), 123);
        assert_eq!(parallel_streamed_reader_buffer_frames(123, 1), 123);
        assert_eq!(parallel_streamed_reader_buffer_frames(123, 4), 492);
        assert_eq!(
            parallel_streamed_reader_buffer_frames(usize::MAX / 2 + 1, 2),
            usize::MAX
        );
    }

    #[test]
    fn parallel_active_needed_window_reports_min_and_active_count() {
        // Underpins both the prune floor (`keep_from`) and the active-worker
        // budget multiplier (P1). Retired workers (`usize::MAX`) are excluded from
        // both the min and the count.
        let all_retired = [
            Arc::new(AtomicUsize::new(usize::MAX)),
            Arc::new(AtomicUsize::new(usize::MAX)),
        ];
        assert_eq!(parallel_active_needed_window(&all_retired), None);
        assert_eq!(minimum_parallel_needed_frame(&all_retired), None);

        let mixed = [
            Arc::new(AtomicUsize::new(usize::MAX)), // retired worker 0
            Arc::new(AtomicUsize::new(500)),
            Arc::new(AtomicUsize::new(100)),
            Arc::new(AtomicUsize::new(usize::MAX)), // retired worker 3
        ];
        assert_eq!(parallel_active_needed_window(&mixed), Some((100, 2)));
        // `minimum_parallel_needed_frame` must still return just the floor so the
        // existing prune callers are unchanged.
        assert_eq!(minimum_parallel_needed_frame(&mixed), Some(100));
    }

    #[test]
    fn parallel_reader_buffer_frames_uses_byte_budget_not_old_512_cap() {
        // P1: after the P2/P3 store reduction a frame is small, so the old hard
        // 512-frame cap truncated the byte budget badly (here several thousand
        // frames fit in 384 MiB at 240p luma-only, yet only 512 were allowed). The
        // per-worker window must now follow the byte budget, capped only by the
        // much larger sanity backstop.
        let video_details = *decoder().get_video_details();
        let opts = standard_options();
        let reducer = parallel_frame_reducer::<u8>(opts, &video_details);
        let per_worker = parallel_reader_buffer_frames::<u8>(opts, &video_details, &reducer);

        assert!(
            per_worker > 512,
            "per-worker budget {per_worker} should exceed the retired 512-frame cap"
        );
        assert!(per_worker <= PARALLEL_READER_MAX_BUFFER_FRAMES);
        // It tracks the byte budget: equals `384 MiB / per-frame cost` here, since
        // that dominates both the per-worker minimum and the 16k backstop.
        let frame_bytes = estimated_frame_bytes(&video_details, &reducer)
            .saturating_add(PARALLEL_READER_FRAME_ENTRY_OVERHEAD_BYTES)
            .max(1);
        assert_eq!(per_worker, PARALLEL_READER_TARGET_BUFFER_BYTES / frame_bytes);
    }

    #[test]
    fn parallel_frame_reducer_gate_keys_off_tuning_not_speed() {
        use crate::analyze::ParallelFrameReducer;

        // Fast + similarity disabled => drop chroma AND pre-downscale (>240p).
        let fast = ParallelFrameReducer::<u8>::new((1920, 1080), SceneDetectionSpeed::Fast, false, false);
        assert!(fast.drops_chroma());
        assert_eq!(fast.scale_factor(), Some(8));
        assert!(fast.is_prescaled());

        // forward-similarity on => keep chroma and never pre-downscale, even Fast.
        let fast_forward =
            ParallelFrameReducer::<u8>::new((1920, 1080), SceneDetectionSpeed::Fast, true, false);
        assert!(!fast_forward.drops_chroma());
        assert!(!fast_forward.is_prescaled());

        // transient-only on => may drop chroma (transient is luma-only) but must
        // not pre-downscale (transient needs full-res luma).
        let fast_transient =
            ParallelFrameReducer::<u8>::new((1920, 1080), SceneDetectionSpeed::Fast, false, true);
        assert!(fast_transient.drops_chroma());
        assert!(!fast_transient.is_prescaled());

        // Standard => never pre-downscale; chroma dropped when forward off.
        let standard =
            ParallelFrameReducer::<u8>::new((1920, 1080), SceneDetectionSpeed::Standard, false, false);
        assert!(standard.drops_chroma());
        assert!(!standard.is_prescaled());

        // <=240p Fast => no downscale factor, so chroma-drop only.
        let small_fast =
            ParallelFrameReducer::<u8>::new((352, 240), SceneDetectionSpeed::Fast, false, false);
        assert!(!small_fast.is_prescaled());
        assert!(small_fast.drops_chroma());
    }
}
