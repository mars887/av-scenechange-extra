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

pub use crate::analyze::{
    ForwardSimilarityCandidate,
    ForwardSimilarityCandidateDecision,
    SceneChangeDetector,
    ScenecutDecision,
    ScenecutResult,
};

const FRAME_PREFETCH_DEPTH: usize = 8;
/// Version marker for diagnostics fields emitted by this fork.
pub const DIAGNOSTICS_VERSION: &str = "av-scenechange-extra-forward-diagnostics-v1";

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
                .max(self.tuning.forward_similarity.frames)
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
            importance_cut_relaxed_ratio: None,
            importance_cut_dark_ratio_boost: 0.0,
            importance_cut_dark_min_ratio: 0.0,
            importance_cut_dark_luma_low_8bit: 25.0,
            importance_cut_dark_luma_high_8bit: 60.0,
            importance_cut_relaxed_min_cost_ratio: 0.0,
            importance_cut_relaxed_max_previous_ratio: 2.2,
            importance_cut_min_me_bad_ratio: 0.0,
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
            importance_cut_max_luma_8bit: Some(60.0),
            importance_cut_relaxed_ratio: Some(3.0),
            importance_cut_dark_ratio_boost: 0.65,
            importance_cut_dark_min_ratio: 2.35,
            importance_cut_dark_luma_low_8bit: 25.0,
            importance_cut_dark_luma_high_8bit: 60.0,
            importance_cut_relaxed_min_cost_ratio: 0.08,
            importance_cut_relaxed_max_previous_ratio: 2.2,
            importance_cut_min_me_bad_ratio: 0.15,
            forward_similarity: ForwardSimilarityOptions {
                enabled: true,
                frames: 80,
                min_offset: 4,
                threshold_8bit: 6.0,
                mask_percent: 0.20,
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
    /// Minimum future offset before a frame can be accepted as a return. This
    /// avoids suppressing hard cuts just because one of the next few frames is
    /// still visually similar to the previous scene.
    pub min_offset: usize,
    /// Maximum luma delta, in 8-bit units, considered a return to the previous
    /// scene. Uses masked block comparison when `mask_percent` is non-zero.
    pub threshold_8bit: f64,
    /// Fraction of most volatile blocks to mask when checking the return.
    pub mask_percent: f64,
    /// Only accept a return frame if there is also a plausible future cut
    /// candidate before that return. This keeps A-B-A suppression
    /// segment-aware while allowing the best matching return frame to be a
    /// stable interior frame after the B->A boundary.
    pub require_return_candidate: bool,
    /// Suppress additional cuts until the detected return frame.
    pub suppress_inside: bool,
}

impl Default for ForwardSimilarityOptions {
    #[inline]
    fn default() -> Self {
        Self {
            enabled: false,
            frames: 0,
            min_offset: 2,
            threshold_8bit: 6.0,
            mask_percent: 0.0,
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
    let effective_lookahead = opts.effective_lookahead_distance();
    let transient_history = if opts.tuning.transient_similarity.enabled {
        opts.tuning.transient_similarity.frames
    } else {
        0
    };
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
            let mut frame_queue = BTreeMap::new();
            let mut keyframes = BTreeSet::new();
            keyframes.insert(0);
            let mut scores = BTreeMap::new();

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
                        .range(frameno.saturating_sub(transient_history + 1)..frameno)
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

                let remove_before = frameno.saturating_sub(transient_history + 1);
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
                if let Some(frame_limit) = frame_limit
                    && frameno == frame_limit
                {
                    break;
                }
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
