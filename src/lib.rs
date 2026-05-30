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
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        mpsc::{channel, sync_channel},
    },
    thread,
    time::Instant,
};

pub use av_decoders::{self, Decoder};
pub use num_rational::Rational32;
use v_frame::pixel::Pixel;

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
const FINALIZED_BATCH_MIN_FRAMES: usize = 16;
/// Version marker for diagnostics fields emitted by this fork.
pub const DIAGNOSTICS_VERSION: &str = "av-scenechange-extra-forward-postprocess-diagnostics-v1";

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

/// A finalized scene-detection frame emitted by the streaming callback.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FinalizedDetectionFrame {
    /// The 0-indexed frame number.
    pub frame: usize,
    /// The score for this frame, if the detector evaluated one.
    pub score: Option<ScenecutResult>,
    /// Whether this frame is a scene-change/keyframe after postprocessing.
    pub is_scene_change: bool,
}

/// A contiguous finalized prefix range from scene detection.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FinalizedDetectionBatch {
    /// Inclusive start of the finalized range.
    pub start_frame: usize,
    /// Exclusive end of the finalized range.
    pub end_frame: usize,
    /// Finalized frames in the range that have either a score or scene
    /// boundary.
    pub frames: Vec<FinalizedDetectionFrame>,
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

    Ok(SceneChangeDetector::new(
        (video_details.width, video_details.height),
        video_details.bit_depth,
        video_details.frame_rate.recip(),
        video_details.chroma_sampling,
        opts.effective_lookahead_distance(),
        opts.analysis_speed,
        opts.tuning,
        opts.min_scenecut_distance.unwrap_or(0),
        opts.max_scenecut_distance.unwrap_or(u32::MAX as usize),
    ))
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
    detect_scene_changes_with_finalized_callback::<T>(
        dec,
        opts,
        frame_limit,
        progress_callback,
        None,
    )
}

/// Runs through a y4m video clip, detecting scene changes and optionally
/// emitting finalized prefix batches while detection is still running.
///
/// The finalized callback receives contiguous ranges that are far enough behind
/// the detector lookahead that subsequent frames cannot change their scores or
/// keyframe decisions. The emitted batches contain enough data to reconstruct
/// the returned `scene_changes` and `scores`; `speed` is only available at the
/// end.
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
pub fn detect_scene_changes_with_finalized_callback<T: Pixel>(
    dec: &mut Decoder,
    opts: DetectionOptions,
    frame_limit: Option<usize>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
    finalized_callback: Option<&dyn Fn(FinalizedDetectionBatch)>,
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
    let (finalized_tx, finalized_rx) = if finalized_callback.is_some() {
        let (tx, rx) = channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let detection_handle = {
        let progress_tx = progress_tx;
        let finalized_tx = finalized_tx;
        thread::spawn(move || -> anyhow::Result<DetectionResults> {
            let mut detector = detector;
            let mut frame_queue = BTreeMap::new();
            let mut keyframes = BTreeSet::new();
            keyframes.insert(0);
            let mut scores = BTreeMap::new();
            let mut finalized_until = 0usize;
            let finalized_lag = finalized_prefix_lag(opts);

            let start_time = Instant::now();
            let mut frameno = 0usize;
            loop {
                let mut next_input_frameno =
                    frame_queue.keys().last().copied().map_or(0, |key| key + 1);
                let max_needed =
                    (frameno + effective_lookahead + 1).min(frame_limit.unwrap_or(usize::MAX));

                while next_input_frameno < max_needed {
                    match frame_rx.recv() {
                        Ok(frame) => {
                            frame_queue.insert(next_input_frameno, frame);
                            next_input_frameno += 1;
                        }
                        Err(_) => break,
                    }
                }

                let frame_set_start = frameno.saturating_sub(1);
                let frame_set = frame_queue
                    .range(frame_set_start..)
                    .map(|(_, frame)| frame)
                    .take(effective_lookahead + 2)
                    .collect::<Vec<_>>();
                if frame_set.len() < 2 {
                    break;
                }
                if frameno == 0 {
                    keyframes.insert(frameno);
                } else {
                    let previous_frame_set = frame_queue
                        .range(frameno.saturating_sub(frame_history.saturating_add(1))..frameno)
                        .map(|(_, frame)| frame)
                        .collect::<Vec<_>>();
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

                let remove_before = frameno.saturating_sub(frame_history.saturating_add(1));
                while frame_queue
                    .keys()
                    .next()
                    .is_some_and(|&oldest| oldest < remove_before)
                {
                    let oldest = *frame_queue
                        .keys()
                        .next()
                        .expect("oldest frame should exist");
                    frame_queue.remove(&oldest);
                }

                frameno += 1;
                if let Some(ref progress_tx) = progress_tx {
                    let _ = progress_tx.send((frameno, keyframes.len()));
                }
                if let Some(ref finalized_tx) = finalized_tx {
                    let end_frame = frameno.saturating_sub(finalized_lag);
                    send_finalized_detection_batch(
                        opts.tuning.forward_similarity,
                        &keyframes,
                        &scores,
                        &mut finalized_until,
                        end_frame,
                        finalized_tx,
                        false,
                    );
                }
                if let Some(frame_limit) = frame_limit
                    && frameno == frame_limit
                {
                    break;
                }
            }

            apply_forward_similarity_postprocess(
                opts.tuning.forward_similarity,
                &mut keyframes,
                &mut scores,
            );
            if let Some(ref finalized_tx) = finalized_tx {
                send_finalized_detection_batch(
                    opts.tuning.forward_similarity,
                    &keyframes,
                    &scores,
                    &mut finalized_until,
                    frameno,
                    finalized_tx,
                    true,
                );
            }

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
        if let (Some(finalized_rx), Some(finalized_fn)) = (&finalized_rx, finalized_callback) {
            while let Ok(batch) = finalized_rx.try_recv() {
                finalized_fn(batch);
            }
        }
    }

    drop(frame_tx);

    if let (Some(progress_rx), Some(progress_fn)) = (&progress_rx, progress_callback) {
        while let Ok((frames, keyframe_count)) = progress_rx.try_recv() {
            progress_fn(frames, keyframe_count);
        }
    }
    if let (Some(finalized_rx), Some(finalized_fn)) = (&finalized_rx, finalized_callback) {
        while let Ok(batch) = finalized_rx.try_recv() {
            finalized_fn(batch);
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
    if let (Some(finalized_rx), Some(finalized_fn)) = (&finalized_rx, finalized_callback) {
        while let Ok(batch) = finalized_rx.try_recv() {
            finalized_fn(batch);
        }
    }

    Ok(results)
}

fn finalized_prefix_lag(opts: DetectionOptions) -> usize {
    opts.effective_lookahead_distance()
        .saturating_add(if opts.tuning.forward_similarity.enabled {
            opts.tuning.forward_similarity.window_frames.max(1)
        } else {
            0
        })
        .saturating_add(1)
}

fn forward_similarity_postprocess_enabled(options: ForwardSimilarityOptions) -> bool {
    options.enabled && options.require_return_candidate && options.frames != 0
}

fn send_finalized_detection_batch(
    forward_similarity: ForwardSimilarityOptions,
    keyframes: &BTreeSet<usize>,
    scores: &BTreeMap<usize, ScenecutResult>,
    finalized_until: &mut usize,
    end_frame: usize,
    finalized_tx: &std::sync::mpsc::Sender<FinalizedDetectionBatch>,
    force: bool,
) {
    if end_frame <= *finalized_until {
        return;
    }
    if !force && end_frame - *finalized_until < FINALIZED_BATCH_MIN_FRAMES {
        return;
    }

    let start_frame = *finalized_until;
    let postprocessed = if forward_similarity_postprocess_enabled(forward_similarity) {
        let mut keyframes = keyframes.clone();
        let mut scores = scores.clone();
        apply_forward_similarity_postprocess(forward_similarity, &mut keyframes, &mut scores);
        Some((keyframes, scores))
    } else {
        None
    };
    let (keyframes, scores) = match &postprocessed {
        Some((keyframes, scores)) => (keyframes, scores),
        None => (keyframes, scores),
    };

    let frames = (start_frame..end_frame)
        .filter_map(|frame| {
            let score = scores.get(&frame).copied();
            let is_scene_change = keyframes.contains(&frame);
            (score.is_some() || is_scene_change).then_some(FinalizedDetectionFrame {
                frame,
                score,
                is_scene_change,
            })
        })
        .collect();
    *finalized_until = end_frame;

    let _ = finalized_tx.send(FinalizedDetectionBatch {
        start_frame,
        end_frame,
        frames,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(value: f64) -> ScenecutResult {
        let mut score = ScenecutResult::new(value, value, 1.0, 1.0, 64.0);
        score.forward_adjusted_cost = value;
        score.refresh_ratios();
        score
    }

    #[test]
    fn finalized_batch_sends_contiguous_prefix_metadata() {
        let (tx, rx) = channel();
        let mut keyframes = BTreeSet::new();
        keyframes.insert(0);
        keyframes.insert(3);
        let mut scores = BTreeMap::new();
        scores.insert(2, score(0.2));
        scores.insert(3, score(1.4));
        let mut finalized_until = 0;

        send_finalized_detection_batch(
            ForwardSimilarityOptions::default(),
            &keyframes,
            &scores,
            &mut finalized_until,
            4,
            &tx,
            true,
        );

        let batch = rx.recv().expect("batch should be sent");
        assert_eq!(finalized_until, 4);
        assert_eq!(batch.start_frame, 0);
        assert_eq!(batch.end_frame, 4);
        assert_eq!(batch.frames.len(), 3);
        assert_eq!(batch.frames[0].frame, 0);
        assert!(batch.frames[0].is_scene_change);
        assert!(batch.frames[0].score.is_none());
        assert_eq!(batch.frames[1].frame, 2);
        assert!(batch.frames[1].score.is_some());
        assert!(!batch.frames[1].is_scene_change);
        assert_eq!(batch.frames[2].frame, 3);
        assert!(batch.frames[2].score.is_some());
        assert!(batch.frames[2].is_scene_change);
    }

    #[test]
    fn finalized_batch_waits_for_minimum_size_without_force() {
        let (tx, rx) = channel();
        let mut keyframes = BTreeSet::new();
        keyframes.insert(0);
        let scores = BTreeMap::new();
        let mut finalized_until = 0;

        send_finalized_detection_batch(
            ForwardSimilarityOptions::default(),
            &keyframes,
            &scores,
            &mut finalized_until,
            FINALIZED_BATCH_MIN_FRAMES - 1,
            &tx,
            false,
        );

        assert_eq!(finalized_until, 0);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn finalized_batch_sends_at_minimum_size_without_force() {
        let (tx, rx) = channel();
        let mut keyframes = BTreeSet::new();
        keyframes.insert(0);
        let scores = BTreeMap::new();
        let mut finalized_until = 0;

        send_finalized_detection_batch(
            ForwardSimilarityOptions::default(),
            &keyframes,
            &scores,
            &mut finalized_until,
            FINALIZED_BATCH_MIN_FRAMES,
            &tx,
            false,
        );

        let batch = rx.recv().expect("batch should be sent");
        assert_eq!(finalized_until, FINALIZED_BATCH_MIN_FRAMES);
        assert_eq!(batch.start_frame, 0);
        assert_eq!(batch.end_frame, FINALIZED_BATCH_MIN_FRAMES);
        assert_eq!(batch.frames.len(), 1);
        assert_eq!(batch.frames[0].frame, 0);
        assert!(batch.frames[0].is_scene_change);
    }
}

#[derive(Debug, Clone, Copy)]
struct PostprocessForwardSimilarityMatch {
    return_frame: usize,
    return_candidate_frame: usize,
    delta: f64,
}

fn apply_forward_similarity_postprocess(
    options: ForwardSimilarityOptions,
    keyframes: &mut BTreeSet<usize>,
    scores: &mut BTreeMap<usize, ScenecutResult>,
) {
    if !forward_similarity_postprocess_enabled(options) {
        return;
    }

    let candidates = keyframes
        .iter()
        .copied()
        .filter(|&frame| frame != 0)
        .collect::<Vec<_>>();
    for frame in candidates {
        if !keyframes.contains(&frame) {
            continue;
        }

        let Some(score) = scores.get(&frame).copied() else {
            continue;
        };
        if !matches!(
            score.decision,
            ScenecutDecision::Cut | ScenecutDecision::CutImportance
        ) {
            continue;
        }
        let previous_frame = keyframes.range(..frame).next_back().copied().unwrap_or(0);
        if !analyze::forward_similarity_start_allowed(options, score, frame - previous_frame) {
            continue;
        }

        let Some(similarity_match) =
            forward_similarity_postprocess_match(frame, options, score, scores)
        else {
            continue;
        };

        keyframes.remove(&frame);
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
