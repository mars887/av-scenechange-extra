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
const PARALLEL_READER_MAX_BUFFER_FRAMES: usize = 512;
const PARALLEL_READER_WAIT: Duration = Duration::from_millis(20);
const FRAME_REF_INLINE_CAPACITY: usize = 96;
/// Version marker for diagnostics fields emitted by this fork.
pub const DIAGNOSTICS_VERSION: &str = "av-scenechange-extra-forward-postprocess-diagnostics-v1";

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

            apply_forward_similarity_postprocess(
                opts.tuning.forward_similarity,
                &mut keyframes,
                &mut scores,
            );

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
    let Some(frame_count) = frame_limit.or(video_details.total_frames) else {
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
    let max_buffered_frames = parallel_reader_buffer_frames(opts, &video_details);
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
                    let result = run_parallel_worker::<T>(
                        worker,
                        start_frame,
                        frame_count,
                        &video_details,
                        opts,
                        worker_store,
                        worker_tx.clone(),
                        worker_stop,
                        worker_needed_from,
                        worker_frame_request_tx,
                        use_worker_cost_parallelism,
                    );
                    let message = match result {
                        Ok(frame_count) => ParallelWorkerMessage::Done {
                            worker,
                            frame_count,
                            error: None,
                        },
                        Err(error) => ParallelWorkerMessage::Done {
                            worker,
                            frame_count: start_frame,
                            error: Some(error.to_string()),
                        },
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
            )
        } else {
            read_parallel_streamed_frames::<T>(
                dec,
                frame_count,
                &store,
                &needed_from,
                max_buffered_frames,
                &progress_rx,
                progress_callback,
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

        let mut results = reconcile_handle
            .join()
            .map_err(|_| anyhow::anyhow!("scene detection reconciliation thread panicked"))??;
        drain_parallel_progress(&progress_rx, progress_callback);

        for stop in &stop_after {
            stop.store(0, Ordering::Release);
        }

        for handle in worker_handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("scene detection worker thread panicked"))?;
        }
        drain_parallel_progress(&progress_rx, progress_callback);

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
    let Some(frame_count) = frame_limit.or(video_details
        .total_frames
        .map(|total_frames| total_frames.saturating_sub(frame_start)))
    else {
        return detect_scene_changes::<T>(dec, opts, frame_limit, progress_callback);
    };
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
                    let result = run_parallel_indexed_decoder_worker::<T, F>(
                        worker,
                        start_frame,
                        frame_start,
                        frame_count,
                        &video_details,
                        opts,
                        worker_tx.clone(),
                        worker_stop,
                        make_decoder,
                        use_worker_cost_parallelism,
                    );
                    let message = match result {
                        Ok(frame_count) => ParallelWorkerMessage::Done {
                            worker,
                            frame_count,
                            error: None,
                        },
                        Err(error) => ParallelWorkerMessage::Done {
                            worker,
                            frame_count: start_frame,
                            error: Some(error.to_string()),
                        },
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
        let mut results = reconcile_handle
            .join()
            .map_err(|_| anyhow::anyhow!("scene detection reconciliation thread panicked"))??;
        drain_parallel_progress(&progress_rx, progress_callback);

        for stop in &stop_after {
            stop.store(0, Ordering::Release);
        }
        for handle in worker_handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("scene detection worker thread panicked"))?;
        }
        drain_parallel_progress(&progress_rx, progress_callback);

        results.speed = results.frame_count as f64 / start_time.elapsed().as_secs_f64();
        if let Some(progress_fn) = progress_callback {
            progress_fn(results.frame_count, results.scene_changes.len());
        }
        Ok(results)
    })
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
    max_buffered_frames: usize,
    progress_rx: &Option<Receiver<(usize, usize)>>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> anyhow::Result<usize> {
    let mut produced = 0usize;
    loop {
        if produced == frame_count {
            break Ok(produced);
        }
        wait_for_parallel_reader_capacity(
            store,
            needed_from,
            max_buffered_frames,
            progress_rx,
            progress_callback,
        )?;
        match dec.read_video_frame() {
            Ok(frame) => {
                store.push(produced, Arc::new(frame));
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
    frame_count: usize,
    store: &SharedFrameStore<T>,
    needed_from: &[Arc<AtomicUsize>],
    frame_request_rx: Receiver<usize>,
    worker_handles: &[thread::ScopedJoinHandle<'scope, ()>],
    reconcile_handle: &thread::ScopedJoinHandle<'scope, anyhow::Result<DetectionResults>>,
    progress_rx: &Option<Receiver<(usize, usize)>>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> anyhow::Result<usize> {
    let mut loaded = BTreeSet::new();
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
                if frame >= frame_count || !loaded.insert(frame) {
                    continue;
                }
                match dec.get_video_frame(frame) {
                    Ok(data) => {
                        store.push(frame, Arc::new(data));
                        prune_parallel_indexed_frame_store(store, needed_from);
                    }
                    Err(av_decoders::DecoderError::EndOfFile) => {
                        store.finish(frame);
                        return Ok(frame);
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
    );
    Ok(frame_count)
}

fn parallel_chunk_starts(frame_count: usize, workers: usize) -> Vec<usize> {
    if workers <= 1 || frame_count == 0 {
        return vec![0];
    }
    let mut starts = (0..workers)
        .map(|idx| idx * frame_count / workers)
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

fn parallel_reader_buffer_frames(
    opts: DetectionOptions,
    video_details: &av_decoders::VideoDetails,
) -> usize {
    let minimum = opts
        .effective_lookahead_distance()
        .saturating_add(parallel_frame_history(opts))
        .saturating_add(8);
    let frame_bytes = estimated_frame_bytes(video_details).max(1);
    let byte_limited = PARALLEL_READER_TARGET_BUFFER_BYTES / frame_bytes;
    byte_limited
        .max(minimum)
        .min(PARALLEL_READER_MAX_BUFFER_FRAMES.max(minimum))
}

fn estimated_frame_bytes(video_details: &av_decoders::VideoDetails) -> usize {
    let luma_pixels = video_details.width.saturating_mul(video_details.height);
    let chroma_pixels = match video_details.chroma_sampling {
        v_frame::chroma::ChromaSubsampling::Yuv420 => luma_pixels / 2,
        v_frame::chroma::ChromaSubsampling::Yuv422 => luma_pixels,
        v_frame::chroma::ChromaSubsampling::Yuv444 => luma_pixels.saturating_mul(2),
        v_frame::chroma::ChromaSubsampling::Monochrome => 0,
    };
    let bytes_per_sample = video_details.bit_depth.div_ceil(8).max(1);
    luma_pixels
        .saturating_add(chroma_pixels)
        .saturating_mul(bytes_per_sample)
}

fn wait_for_parallel_reader_capacity<T: Pixel>(
    store: &SharedFrameStore<T>,
    needed_from: &[Arc<AtomicUsize>],
    max_buffered_frames: usize,
    progress_rx: &Option<Receiver<(usize, usize)>>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> anyhow::Result<()> {
    loop {
        drain_parallel_progress(progress_rx, progress_callback);
        let Some(keep_from) = minimum_parallel_needed_frame(needed_from) else {
            return Ok(());
        };
        store.prune_before(keep_from);
        let produced = store.produced_or_error()?;
        if produced.saturating_sub(keep_from) <= max_buffered_frames {
            return Ok(());
        }
        store.wait_for_change(PARALLEL_READER_WAIT);
    }
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
    needed_from
        .iter()
        .map(|needed| needed.load(Ordering::Acquire))
        .filter(|&frame| frame != usize::MAX)
        .min()
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
    frame_limit: usize,
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
    let effective_lookahead = opts.effective_lookahead_distance();
    let frame_history = parallel_frame_history(opts);
    assert!(effective_lookahead >= 1);

    let initial_fetch_start = parallel_initial_fetch_start(start_frame, opts);
    let mut detector =
        new_detector_from_video_details::<T>(video_details, opts, use_cost_parallelism);
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
            let frame = source.get_video_frame(source_frame).map_err(|error| {
                anyhow::anyhow!("worker {worker} failed to read frame {source_frame}: {error}")
            })?;
            frame_queue.push_next(next_input_frameno, Arc::new(frame));
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
    let mut analyzed_frames = vec![false; frame_limit];
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
                actual_frame_limit = frame_count.min(frame_limit);
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

    apply_forward_similarity_postprocess(
        opts.tuning.forward_similarity,
        &mut keyframes,
        &mut scores,
    );

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
}
