use std::{
    cmp,
    collections::BTreeMap,
    num::{NonZeroU8, NonZeroUsize},
    sync::Arc,
};

use log::debug;
use num_rational::Rational32;
use v_frame::{chroma::ChromaSubsampling, frame::Frame, pixel::Pixel, plane::Plane};

use self::fast::{FAST_THRESHOLD, detect_scale_factor};
use crate::{
    DetectionTuning,
    FastThresholdScale,
    ForwardSimilarityOptions,
    ImportanceAggregation,
    ImportanceThresholdMode,
    SceneDetectionSpeed,
    TransientSimilarityOptions,
    data::{
        motion::RefMEStats,
        plane::{downscale, downscale_in_place},
        sad::sad_plane,
    },
};

mod fast;
mod importance;
mod inter;
mod intra;
mod standard;

const FORWARD_SIMILARITY_DIAGNOSTIC_SLOTS: usize = 8;

#[cfg(feature = "bench-internals")]
pub use self::{
    importance::estimate_importance_block_difference,
    inter::estimate_inter_costs,
    intra::estimate_intra_costs,
};

/// Fast integer division where divisor is a nonzero power of 2
pub(crate) fn fast_idiv(n: usize, d: NonZeroUsize) -> usize {
    debug_assert!(d.is_power_of_two());

    n >> d.trailing_zeros()
}

struct ScaleFunction<T: Pixel> {
    downscale_in_place: fn(/* &self: */ &Plane<T>, /* in_plane: */ &mut Plane<T>),
    downscale: fn(/* &self: */ &Plane<T>, /* bit_depth */ NonZeroU8) -> Plane<T>,
    factor: NonZeroUsize,
}

impl<T: Pixel> ScaleFunction<T> {
    fn from_scale<const SCALE: usize>() -> Self {
        assert!(
            SCALE.is_power_of_two(),
            "Scaling factor needs to be a nonzero power of two"
        );

        Self {
            downscale: downscale::<T, SCALE>,
            downscale_in_place: downscale_in_place::<T, SCALE>,
            factor: NonZeroUsize::new(SCALE).expect("scale must not be zero"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ForwardSimilarityMatch {
    frame: usize,
    delta: f64,
}

#[derive(Clone, Copy, Debug)]
struct ForwardSimilaritySearch {
    accepted: Option<ForwardSimilarityMatch>,
    candidates: [Option<ForwardSimilarityCandidate>; FORWARD_SIMILARITY_DIAGNOSTIC_SLOTS],
    rejected_candidates: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum ForwardSimilarityCandidateDecision {
    Accepted,
    MissingReturnCandidate,
    DeltaAboveThreshold,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct ForwardSimilarityCandidate {
    pub frame: usize,
    pub offset: usize,
    pub delta: f64,
    pub threshold: f64,
    pub candidate_frame: Option<usize>,
    pub candidate_offset: Option<usize>,
    pub decision: ForwardSimilarityCandidateDecision,
}

#[derive(Clone, Debug)]
pub(crate) struct ScenecutAnalysis {
    pub result: ScenecutResult,
    pub importance_blocks: Vec<f64>,
    pub importance_cols: usize,
    pub importance_rows: usize,
}

impl ScenecutAnalysis {
    #[inline]
    fn from_result(result: ScenecutResult) -> Self {
        Self {
            result,
            importance_blocks: Vec::new(),
            importance_cols: 0,
            importance_rows: 0,
        }
    }
}

/// Runs keyframe detection on frames from the lookahead queue.
///
/// This struct is intended for advanced users who need the ability to analyze
/// a small subset of frames at a time, for example in a streaming fashion.
/// Most users will prefer to use `new_detector` and `detect_scene_changes`
/// at the top level of this crate.
pub struct SceneChangeDetector<T: Pixel> {
    // User configuration options
    /// Scenecut detection mode
    scene_detection_mode: SceneDetectionSpeed,
    /// Deque offset for current
    lookahead_offset: usize,
    /// Minimum number of frames between two scenecuts
    min_key_frame_interval: usize,
    /// Maximum number of frames between two scenecuts
    max_key_frame_interval: usize,
    tuning: DetectionTuning,

    // Internal configuration options
    /// Minimum average difference between YUV deltas that will trigger a scene
    /// change.
    threshold: f64,
    /// Width and height of the unscaled frame
    resolution: (usize, usize),
    /// The bit depth of the video.
    bit_depth: usize,
    /// The frame rate of the video.
    frame_rate: Rational32,
    /// The chroma subsampling of the video.
    chroma_sampling: ChromaSubsampling,
    /// Number of pixels in scaled frame for fast mode
    scaled_pixels: usize,
    /// Downscaling function for fast scene detection
    scale_func: Option<ScaleFunction<T>>,

    // Internal data structures
    /// Start deque offset based on lookahead
    deque_offset: usize,
    /// Frame buffer for scaled frames
    downscaled_frame_buffer: Option<[Plane<T>; 2]>,
    /// Scenechange results for adaptive threshold
    score_deque: Vec<ScenecutAnalysis>,
    /// Suppresses additional cuts while an A-B-A transient is active.
    forward_suppress_until: Option<usize>,
    /// Reused per-block deltas for masked forward/transient similarity.
    similarity_block_deltas: Vec<f64>,
    /// Reused mask for spatially capped volatile blocks.
    similarity_block_mask: Vec<bool>,
    /// Reused block indices for spatially capped volatile-block selection.
    similarity_block_indices: Vec<usize>,
    /// Reused per-region masked block counts.
    similarity_region_counts: Vec<usize>,
    /// Temporary buffer used by `estimate_intra_costs`.
    /// We store it on the struct so we only need to allocate it once.
    temp_plane: Option<Plane<T>>,
    /// Buffer for `FrameMEStats` for cost scenecut
    frame_me_stats_buffer: Option<RefMEStats>,

    /// Calculated intra costs for each input frame.
    /// These can be cached for reuse by advanced API users.
    /// Caching will occur if this is not `None`.
    pub intra_costs: Option<BTreeMap<usize, Box<[u32]>>>,
}

impl<T: Pixel> SceneChangeDetector<T> {
    /// Creates a new instance of the `SceneChangeDetector`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::missing_panics_doc)]
    #[inline]
    #[must_use]
    pub fn new(
        resolution: (usize, usize),
        bit_depth: usize,
        frame_rate: Rational32,
        chroma_sampling: ChromaSubsampling,
        lookahead_distance: usize,
        scene_detection_mode: SceneDetectionSpeed,
        tuning: DetectionTuning,
        min_key_frame_interval: usize,
        max_key_frame_interval: usize,
    ) -> Self {
        // Downscaling function for fast scene detection
        let scale_func = detect_scale_factor(resolution, scene_detection_mode);

        // Set lookahead offset to 5 if normal lookahead available
        let lookahead_offset = if lookahead_distance >= 5 { 5 } else { 0 };
        let deque_offset = lookahead_offset;

        let score_deque = Vec::with_capacity(5 + lookahead_distance);

        // Downscaling factor for fast scenedetect (is currently always a power of 2)
        let factor = scale_func.as_ref().map_or(
            NonZeroUsize::new(1).expect("constant should not panic"),
            |x| x.factor,
        );

        let pixels = if scene_detection_mode == SceneDetectionSpeed::Fast {
            fast_idiv(resolution.1, factor) * fast_idiv(resolution.0, factor)
        } else {
            1
        };

        let threshold = match tuning.fast_threshold_scale {
            FastThresholdScale::Legacy => FAST_THRESHOLD * (bit_depth as f64) / 8.0,
            FastThresholdScale::SampleRange => {
                tuning.fast_threshold_8bit * sample_range_scale(bit_depth)
            }
        };

        Self {
            threshold,
            scene_detection_mode,
            tuning,
            scale_func,
            lookahead_offset,
            deque_offset,
            score_deque,
            forward_suppress_until: None,
            similarity_block_deltas: Vec::new(),
            similarity_block_mask: Vec::new(),
            similarity_block_indices: Vec::new(),
            similarity_region_counts: Vec::new(),
            scaled_pixels: pixels,
            bit_depth,
            frame_rate,
            chroma_sampling,
            min_key_frame_interval,
            max_key_frame_interval,
            downscaled_frame_buffer: None,
            resolution,
            temp_plane: None,
            frame_me_stats_buffer: None,
            intra_costs: None,
        }
    }

    /// Enables caching of intra costs. For advanced API users.
    #[inline]
    pub fn enable_cache(&mut self) {
        if self.intra_costs.is_none() {
            self.intra_costs = Some(BTreeMap::new());
        }
    }

    /// Runs keyframe detection on the next frame in the lookahead queue.
    ///
    /// This function requires that a subset of input frames
    /// is passed to it in order, and that `keyframes` is only
    /// updated from this method. `input_frameno` should correspond
    /// to the second frame in `frame_set`.
    ///
    /// This will gracefully handle the first frame in the video as well.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(input_frameno))
    )]
    #[inline]
    pub fn analyze_next_frame(
        &mut self,
        frame_set: &[&Arc<Frame<T>>],
        input_frameno: usize,
        previous_keyframe: usize,
    ) -> (bool, Option<ScenecutResult>) {
        self.analyze_next_frame_with_history(frame_set, &[], input_frameno, previous_keyframe)
    }

    pub(crate) fn analyze_next_frame_with_history(
        &mut self,
        frame_set: &[&Arc<Frame<T>>],
        previous_frame_set: &[&Arc<Frame<T>>],
        input_frameno: usize,
        previous_keyframe: usize,
    ) -> (bool, Option<ScenecutResult>) {
        // Use score deque for adaptive threshold for scene cut
        // Declare score_deque offset based on lookahead  for scene change scores

        // Find the distance to the previous keyframe.
        let distance = input_frameno - previous_keyframe;

        if frame_set.len() <= self.lookahead_offset {
            // Don't insert keyframes in the last few frames of the video
            // This is basically a scene flash and a waste of bits
            return (false, None);
        }

        if self.scene_detection_mode == SceneDetectionSpeed::None {
            if self.handle_min_max_intervals(distance) == Some(true) {
                return (true, None);
            };
            return (false, None);
        }

        // Initialization of score deque
        // based on frame set length
        if self.deque_offset > 0
            && frame_set.len() > self.deque_offset + 1
            && self.score_deque.is_empty()
        {
            self.initialize_score_deque(frame_set, input_frameno, self.deque_offset);
        } else if self.score_deque.is_empty() {
            self.initialize_score_deque(frame_set, input_frameno, frame_set.len() - 1);

            self.deque_offset = frame_set.len() - 2;
        }
        // Running single frame comparison and adding it to deque
        // Decrease deque offset if there is no new frames
        if frame_set.len() > self.deque_offset + 1 {
            self.run_comparison(
                frame_set[self.deque_offset],
                frame_set[self.deque_offset + 1],
                input_frameno + self.deque_offset,
            );
        } else {
            self.deque_offset -= 1;
        }

        // Adaptive scenecut check
        let (mut scenecut, mut score) =
            self.adaptive_scenecut(frame_set, previous_frame_set, input_frameno, distance);
        if let Some(interval_decision) = self.handle_min_max_intervals(distance) {
            scenecut = interval_decision;
            score.decision = if interval_decision {
                ScenecutDecision::ForcedMaxDistance
            } else {
                ScenecutDecision::SuppressedMinDistance
            };
            if self.deque_offset < self.score_deque.len() {
                self.score_deque[self.deque_offset].result = score;
            }
        }
        debug!(
            "[SC-Detect] Frame {}: Raw={:5.1}  ImpBl={:5.1}/{:.1}  Bwd={:5.1}  Fwd={:5.1}  \
             Th={:.1}  {:?}",
            input_frameno,
            score.inter_cost,
            score.imp_block_cost,
            score.imp_block_threshold,
            score.backward_adjusted_cost,
            score.forward_adjusted_cost,
            score.threshold,
            score.decision,
        );

        // Keep score deque of 5 backward frames
        // and forward frames of length of lookahead offset
        if self.score_deque.len() > 5 + self.lookahead_offset {
            self.score_deque.pop();
        }

        (scenecut, Some(score))
    }

    fn handle_min_max_intervals(&self, distance: usize) -> Option<bool> {
        // Handle minimum and maximum keyframe intervals.
        if distance < self.min_key_frame_interval {
            return Some(false);
        }
        if distance >= self.max_key_frame_interval {
            return Some(true);
        }
        None
    }

    // Initially fill score deque with frame scores
    fn initialize_score_deque(
        &mut self,
        frame_set: &[&Arc<Frame<T>>],
        input_frameno: usize,
        init_len: usize,
    ) {
        for x in 0..init_len {
            self.run_comparison(frame_set[x], frame_set[x + 1], input_frameno + x);
        }
    }

    /// Runs scene change comparison between 2 given frames
    /// Insert result to start of score deque
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(input_frameno))
    )]
    fn run_comparison(
        &mut self,
        frame1: &Arc<Frame<T>>,
        frame2: &Arc<Frame<T>>,
        input_frameno: usize,
    ) {
        let mut analysis = match self.scene_detection_mode {
            SceneDetectionSpeed::Fast => {
                ScenecutAnalysis::from_result(self.fast_scenecut(frame1, frame2))
            }
            SceneDetectionSpeed::Standard | SceneDetectionSpeed::High => {
                self.cost_scenecut(frame1, frame2, input_frameno)
            }
            _ => unreachable!(),
        };

        // Subtract the highest metric value of surrounding frames from the current one.
        // It makes the peaks in the metric more distinct.
        if matches!(
            self.scene_detection_mode,
            SceneDetectionSpeed::Standard | SceneDetectionSpeed::High
        ) && self.deque_offset > 0
        {
            if input_frameno == 1 {
                // Accounts for the second frame not having a score to adjust against.
                // It should always be 0 because the first frame of the video is always a
                // keyframe.
                analysis.result.backward_adjusted_cost = 0.0;
            } else {
                let mut adjusted_cost = f64::MAX;
                for other_cost in self
                    .score_deque
                    .iter()
                    .take(self.deque_offset)
                    .map(|i| i.result.inter_cost)
                {
                    let this_cost = analysis.result.inter_cost - other_cost;
                    if this_cost < adjusted_cost {
                        adjusted_cost = this_cost;
                    }
                    if adjusted_cost < 0.0 {
                        adjusted_cost = 0.0;
                        break;
                    }
                }
                analysis.result.backward_adjusted_cost = adjusted_cost;
            }
            if !self.score_deque.is_empty() {
                for i in 0..cmp::min(self.deque_offset, self.score_deque.len()) {
                    let adjusted_cost =
                        self.score_deque[i].result.inter_cost - analysis.result.inter_cost;
                    if i == 0 || adjusted_cost < self.score_deque[i].result.forward_adjusted_cost {
                        self.score_deque[i].result.forward_adjusted_cost = adjusted_cost;
                    }
                    if self.score_deque[i].result.forward_adjusted_cost < 0.0 {
                        self.score_deque[i].result.forward_adjusted_cost = 0.0;
                    }
                }
            }
        }
        self.score_deque.insert(0, analysis);
    }

    /// Compares current scene score to adapted threshold based on previous
    /// scores
    ///
    /// Value of current frame is offset by lookahead, if lookahead >=5
    ///
    /// Returns true if current scene score is higher than adapted threshold
    fn adaptive_scenecut(
        &mut self,
        frame_set: &[&Arc<Frame<T>>],
        previous_frame_set: &[&Arc<Frame<T>>],
        input_frameno: usize,
        previous_scene_len: usize,
    ) -> (bool, ScenecutResult) {
        for idx in 0..self.score_deque.len() {
            self.refresh_importance_metrics(idx);
        }
        let mut score = self.score_deque[self.deque_offset].result;

        // We use the importance block algorithm's cost metrics as a secondary algorithm
        // because, although it struggles in certain scenarios such as
        // finding the end of a pan, it is very good at detecting hard scenecuts
        // or detecting if a pan exists.
        //
        // Because of this, we only consider a frame for a scenechange if
        // the importance block algorithm is over the threshold either on this frame
        // (hard scenecut) or within the past few frames (pan). This helps
        // filter out a few false positives produced by the cost-based
        // algorithm.
        let importance_gate_passed = self.score_deque[self.deque_offset..]
            .iter()
            .any(|analysis| {
                analysis.result.imp_block_cost >= analysis.result.imp_block_threshold
                    || analysis.result.global_imp_block_cost >= analysis.result.imp_block_threshold
            });
        let strong_cut_gate_passed = self
            .tuning
            .strong_cut_ratio
            .is_some_and(|ratio| score.cost_ratio >= ratio);
        if !importance_gate_passed && !strong_cut_gate_passed {
            score.decision = ScenecutDecision::SuppressedImportance;
            self.score_deque[self.deque_offset].result = score;
            return (false, score);
        }

        let cost = score.forward_adjusted_cost;
        let mut scenecut = cost >= score.threshold;
        if cost >= score.threshold {
            let back_deque = &self.score_deque[self.deque_offset + 1..];
            let forward_deque = &self.score_deque[..self.deque_offset];
            let back_over_tr_count = back_deque
                .iter()
                .filter(|analysis| {
                    analysis.result.backward_adjusted_cost >= analysis.result.threshold
                })
                .count();
            let forward_over_tr_count = forward_deque
                .iter()
                .filter(|analysis| {
                    analysis.result.forward_adjusted_cost >= analysis.result.threshold
                })
                .count();

            // Check for scenecut after the flashes
            // No frames over threshold forward
            // and some frames over threshold backward
            let back_count_req = if self.scene_detection_mode == SceneDetectionSpeed::Fast {
                // Fast scenecut is more sensitive to false flash detection,
                // so we want more "evidence" of there being a flash before creating a keyframe.
                2
            } else {
                1
            };
            if forward_over_tr_count == 0 && back_over_tr_count >= back_count_req {
                score.decision = ScenecutDecision::Cut;
                scenecut = true;
            } else if back_over_tr_count == 0
                && forward_over_tr_count == 1
                && forward_deque[0].result.forward_adjusted_cost
                    >= forward_deque[0].result.threshold
            {
                score.decision = ScenecutDecision::Cut;
                scenecut = true;
            } else if back_over_tr_count != 0 || forward_over_tr_count != 0 {
                score.decision = ScenecutDecision::SuppressedFlash;
                scenecut = false;
            } else {
                score.decision = ScenecutDecision::Cut;
                scenecut = true;
            }
        } else {
            score.decision = ScenecutDecision::NoCut;
        }

        if !scenecut
            && matches!(self.scene_detection_mode, SceneDetectionSpeed::High)
            && self.importance_cut_passed(self.deque_offset)
        {
            score.decision = ScenecutDecision::CutImportance;
            scenecut = true;
        }

        if scenecut
            && matches!(score.decision, ScenecutDecision::CutImportance)
            && let Some(delta) = self.two_sided_masked_similarity(previous_frame_set, frame_set)
        {
            score.transient_similarity_score = Some(delta);
            if delta <= self.transient_similarity_threshold(score.avg_luma_8bit) {
                score.decision = ScenecutDecision::SuppressedTransientSimilarity;
                scenecut = false;
            }
        }

        if scenecut && self.is_forward_suppressed(input_frameno) {
            score.decision = ScenecutDecision::SuppressedForwardSimilarity;
            scenecut = false;
        }

        if scenecut
            && forward_similarity_start_allowed(
                self.tuning.forward_similarity,
                score,
                previous_scene_len,
            )
            && let Some(search) =
                self.forward_similarity_search(frame_set, previous_frame_set, input_frameno, score)
        {
            score.forward_similarity_candidates = search.candidates;
            score.forward_similarity_rejected_candidates = search.rejected_candidates;
            if let Some(return_frame) = search.accepted {
                score.decision = ScenecutDecision::SuppressedForwardSimilarity;
                score.forward_return_frame = Some(return_frame.frame);
                score.forward_similarity_score = Some(return_frame.delta);
                if self.tuning.forward_similarity.suppress_inside {
                    self.forward_suppress_until = Some(return_frame.frame);
                }
                scenecut = false;
            }
        }

        self.score_deque[self.deque_offset].result = score;
        (scenecut, score)
    }

    fn importance_cut_passed(&self, index: usize) -> bool {
        let Some(min_ratio) = self.tuning.importance_cut_ratio else {
            return false;
        };
        let Some(current) = self.score_deque.get(index).map(|analysis| analysis.result) else {
            return false;
        };
        if current.imp_block_cost < current.imp_block_threshold
            && current.global_imp_block_cost < current.imp_block_threshold
        {
            return false;
        }
        if let Some(max_luma_8bit) = self.tuning.importance_cut_max_luma_8bit
            && current.avg_luma_8bit > max_luma_8bit
        {
            return false;
        }

        let strict_passed = current.imp_block_ratio >= min_ratio
            && current.cost_ratio >= self.tuning.importance_cut_min_cost_ratio;
        let relaxed_passed = self
            .tuning
            .importance_cut_relaxed_ratio
            .is_some_and(|ratio| {
                let ratio = self.dark_adapted_relaxed_ratio(current.avg_luma_8bit, ratio);
                let previous_ratio = self
                    .score_deque
                    .get(index + 1)
                    .map_or(0.0, |analysis| analysis.result.imp_block_ratio);
                let spatial_passed = current.imp_block_ratio >= ratio;
                let global_passed = current.global_imp_block_ratio >= ratio + 0.35
                    && current.imp_block_ratio >= ratio * 0.75;
                (spatial_passed || global_passed)
                    && current.cost_ratio >= self.tuning.importance_cut_relaxed_min_cost_ratio
                    && previous_ratio <= self.tuning.importance_cut_relaxed_max_previous_ratio
                    && current.me_bad_block_ratio >= self.tuning.importance_cut_min_me_bad_ratio
            });
        if !strict_passed && !relaxed_passed {
            return false;
        }

        self.score_deque
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != index && idx.abs_diff(index) <= 2)
            .all(|(_, analysis)| analysis.result.imp_block_cost <= current.imp_block_cost)
    }

    fn refresh_importance_metrics(&mut self, index: usize) {
        let Some(analysis) = self.score_deque.get(index) else {
            return;
        };
        let global_imp_block_cost = match self.tuning.importance_aggregation {
            ImportanceAggregation::TemporalTopBlocks {
                previous_percent,
                current_percent,
                next_percent,
            }
            | ImportanceAggregation::SpatialTemporalTopBlocks {
                previous_percent,
                current_percent,
                next_percent,
                ..
            } => self
                .temporal_top_importance_score(
                    index,
                    previous_percent,
                    current_percent,
                    next_percent,
                )
                .unwrap_or(analysis.result.imp_block_cost_raw),
            ImportanceAggregation::Mean => analysis.result.imp_block_cost_raw,
        };
        let imp_block_cost = match self.tuning.importance_aggregation {
            ImportanceAggregation::Mean => analysis.result.imp_block_cost_raw,
            ImportanceAggregation::TemporalTopBlocks {
                previous_percent,
                current_percent,
                next_percent,
            } => self
                .temporal_top_importance_score(
                    index,
                    previous_percent,
                    current_percent,
                    next_percent,
                )
                .unwrap_or(analysis.result.imp_block_cost_raw),
            ImportanceAggregation::SpatialTemporalTopBlocks {
                previous_percent,
                current_percent,
                next_percent,
                region_cols,
                region_rows,
            } => self
                .temporal_top_importance_score_spatially_capped(
                    index,
                    previous_percent,
                    current_percent,
                    next_percent,
                    region_cols,
                    region_rows,
                )
                .unwrap_or(analysis.result.imp_block_cost_raw),
        };
        let avg_luma_8bit = analysis.result.avg_luma_8bit;
        let threshold = self.importance_threshold(avg_luma_8bit);
        if let Some(analysis) = self.score_deque.get_mut(index) {
            analysis.result.imp_block_cost = imp_block_cost;
            analysis.result.global_imp_block_cost = global_imp_block_cost;
            analysis.result.imp_block_threshold = threshold;
            analysis.result.refresh_ratios();
        }
    }

    fn dark_adapted_relaxed_ratio(&self, avg_luma_8bit: f64, base_ratio: f64) -> f64 {
        let darkness = smoothstep_darkness(
            avg_luma_8bit,
            self.tuning.importance_cut_dark_luma_low_8bit,
            self.tuning.importance_cut_dark_luma_high_8bit,
        );
        (base_ratio - self.tuning.importance_cut_dark_ratio_boost * darkness)
            .max(self.tuning.importance_cut_dark_min_ratio)
    }

    fn temporal_top_importance_score(
        &self,
        index: usize,
        previous_percent: f64,
        current_percent: f64,
        next_percent: f64,
    ) -> Option<f64> {
        let current = self.score_deque.get(index)?;
        let block_count = current.importance_blocks.len();
        if block_count == 0 {
            return None;
        }

        let mut selected = vec![false; block_count];
        if let Some(previous) = self.score_deque.get(index + 1) {
            mark_top_blocks(&previous.importance_blocks, previous_percent, &mut selected);
        }
        mark_top_blocks(&current.importance_blocks, current_percent, &mut selected);
        if index > 0
            && let Some(next) = self.score_deque.get(index - 1)
        {
            mark_top_blocks(&next.importance_blocks, next_percent, &mut selected);
        }

        let mut total = 0.0;
        let mut selected_count = 0usize;
        for (idx, &is_selected) in selected.iter().enumerate() {
            if is_selected {
                total += current.importance_blocks[idx];
                selected_count += 1;
            }
        }

        (selected_count > 0).then_some(total / selected_count as f64)
    }

    fn temporal_top_importance_score_spatially_capped(
        &self,
        index: usize,
        previous_percent: f64,
        current_percent: f64,
        next_percent: f64,
        region_cols: usize,
        region_rows: usize,
    ) -> Option<f64> {
        let current = self.score_deque.get(index)?;
        let block_count = current.importance_blocks.len();
        if block_count == 0 || current.importance_cols == 0 || current.importance_rows == 0 {
            return None;
        }

        let mut selected = vec![false; block_count];
        if let Some(previous) = self.score_deque.get(index + 1) {
            mark_top_blocks_spatially_capped(
                &previous.importance_blocks,
                previous_percent,
                current.importance_cols,
                current.importance_rows,
                region_cols,
                region_rows,
                &mut selected,
            );
        }
        mark_top_blocks_spatially_capped(
            &current.importance_blocks,
            current_percent,
            current.importance_cols,
            current.importance_rows,
            region_cols,
            region_rows,
            &mut selected,
        );
        if index > 0
            && let Some(next) = self.score_deque.get(index - 1)
        {
            mark_top_blocks_spatially_capped(
                &next.importance_blocks,
                next_percent,
                current.importance_cols,
                current.importance_rows,
                region_cols,
                region_rows,
                &mut selected,
            );
        }

        let mut total = 0.0;
        let mut selected_count = 0usize;
        for (idx, &is_selected) in selected.iter().enumerate() {
            if is_selected {
                total += current.importance_blocks[idx];
                selected_count += 1;
            }
        }

        (selected_count > 0).then_some(total / selected_count as f64)
    }

    pub(crate) fn importance_threshold(&self, avg_luma_8bit: f64) -> f64 {
        let base = match self.tuning.importance_mode {
            ImportanceThresholdMode::Fixed => {
                self.tuning.importance_threshold_8bit * (self.bit_depth as f64) / 8.0
            }
            ImportanceThresholdMode::AdaptiveLuma => {
                self.tuning.importance_threshold_8bit * sample_range_scale(self.bit_depth)
            }
        };
        match self.tuning.importance_mode {
            ImportanceThresholdMode::Fixed => base,
            ImportanceThresholdMode::AdaptiveLuma => {
                let luma_ref = self.tuning.importance_luma_ref_8bit.max(1.0);
                let factor = (avg_luma_8bit.max(0.0) / luma_ref)
                    .sqrt()
                    .clamp(self.tuning.importance_min_factor, 1.0);
                base * factor
            }
        }
    }

    fn is_forward_suppressed(&mut self, input_frameno: usize) -> bool {
        if let Some(until) = self.forward_suppress_until {
            if input_frameno <= until {
                return true;
            }
            self.forward_suppress_until = None;
        }
        false
    }

    fn forward_similarity_search(
        &mut self,
        frame_set: &[&Arc<Frame<T>>],
        previous_frame_set: &[&Arc<Frame<T>>],
        input_frameno: usize,
        score: ScenecutResult,
    ) -> Option<ForwardSimilaritySearch> {
        let options = self.tuning.forward_similarity;
        let ForwardSimilarityOptions {
            enabled,
            frames,
            window_frames,
            min_offset,
            require_return_candidate,
            ..
        } = options;
        if !enabled || frames == 0 || frame_set.len() < 3 {
            return None;
        }

        let window_frames = window_frames.max(1);
        let mut reference_window = previous_frame_set
            .iter()
            .copied()
            .rev()
            .take(window_frames)
            .collect::<Vec<_>>();
        reference_window.reverse();
        if reference_window.is_empty() {
            reference_window.push(frame_set[0]);
        }

        let max_offset = frames
            .saturating_add(window_frames.saturating_sub(1))
            .saturating_add(1)
            .min(frame_set.len().saturating_sub(1));
        let min_offset = min_offset.max(2);
        let threshold_8bit = forward_similarity_threshold_8bit(options, score);
        let mut candidates = [None; FORWARD_SIMILARITY_DIAGNOSTIC_SLOTS];
        let mut accepted = None;
        let mut rejected_candidates = 0usize;
        for offset in min_offset..=max_offset {
            let return_candidate = self.forward_return_candidate(min_offset, offset);

            let Some(post_start_offset) = self.forward_similarity_post_start_offset(
                offset,
                return_candidate,
                reference_window.len(),
            ) else {
                rejected_candidates += 1;
                continue;
            };
            let delta = self.forward_segment_similarity_delta_8bit(
                &reference_window,
                frame_set,
                post_start_offset,
                offset,
                options,
            );
            let decision = if require_return_candidate && return_candidate.is_none() {
                ForwardSimilarityCandidateDecision::MissingReturnCandidate
            } else if delta > threshold_8bit {
                ForwardSimilarityCandidateDecision::DeltaAboveThreshold
            } else {
                ForwardSimilarityCandidateDecision::Accepted
            };
            let candidate = ForwardSimilarityCandidate {
                frame: input_frameno + offset - 1,
                offset,
                delta,
                threshold: threshold_8bit,
                candidate_frame: return_candidate
                    .map(|candidate_offset| input_frameno + candidate_offset - 1),
                candidate_offset: return_candidate,
                decision,
            };
            record_forward_similarity_candidate(&mut candidates, candidate);
            if decision == ForwardSimilarityCandidateDecision::Accepted {
                accepted = Some(ForwardSimilarityMatch {
                    frame: candidate.frame,
                    delta,
                });
                break;
            }
            rejected_candidates += 1;
        }
        Some(ForwardSimilaritySearch {
            accepted,
            candidates,
            rejected_candidates,
        })
    }

    fn forward_similarity_post_start_offset(
        &self,
        return_offset: usize,
        return_candidate: Option<usize>,
        comparison_frames: usize,
    ) -> Option<usize> {
        if comparison_frames == 0 || return_offset < comparison_frames {
            return None;
        }

        let post_start_offset = return_offset + 1 - comparison_frames;
        if post_start_offset == 0 {
            return None;
        }
        if let Some(return_candidate) = return_candidate
            && post_start_offset < return_candidate
        {
            return None;
        }
        Some(post_start_offset)
    }

    fn forward_segment_similarity_delta_8bit(
        &mut self,
        reference_window: &[&Arc<Frame<T>>],
        frame_set: &[&Arc<Frame<T>>],
        post_start_offset: usize,
        post_end_offset: usize,
        options: ForwardSimilarityOptions,
    ) -> f64 {
        let comparison_frames = reference_window
            .len()
            .min(post_end_offset + 1 - post_start_offset);
        if comparison_frames == 0 {
            return f64::MAX;
        }

        let reference_start = reference_window.len() - comparison_frames;
        let mut lower_bound = 0.0;
        for idx in 0..comparison_frames {
            lower_bound += self.frame_similarity_lower_bound_8bit(
                reference_window[reference_start + idx],
                frame_set[post_start_offset + idx],
                options,
            );
        }
        lower_bound /= comparison_frames as f64;
        if options.mask_percent <= 0.0 || lower_bound > options.threshold_8bit {
            return lower_bound;
        }

        let mut delta = 0.0;
        for idx in 0..comparison_frames {
            delta += self.frame_similarity_delta_8bit(
                reference_window[reference_start + idx],
                frame_set[post_start_offset + idx],
                options,
            );
        }
        delta / comparison_frames as f64
    }

    fn forward_return_candidate(&self, min_offset: usize, return_offset: usize) -> Option<usize> {
        (min_offset..=return_offset)
            .rev()
            .find(|&candidate_offset| self.forward_return_candidate_passed(candidate_offset))
    }

    fn forward_return_candidate_passed(&self, offset: usize) -> bool {
        let frame_offset = offset.saturating_sub(1);
        if frame_offset == 0 || frame_offset > self.deque_offset {
            return false;
        }
        let index = self.deque_offset - frame_offset;
        self.score_deque.get(index).is_some_and(|analysis| {
            analysis.result.forward_adjusted_cost >= analysis.result.threshold
                || self.importance_cut_passed(index)
        })
    }

    fn two_sided_masked_similarity(
        &mut self,
        previous_frame_set: &[&Arc<Frame<T>>],
        frame_set: &[&Arc<Frame<T>>],
    ) -> Option<f64> {
        let TransientSimilarityOptions {
            enabled,
            frames,
            mask_percent,
            ..
        } = self.tuning.transient_similarity;
        if !enabled || frames == 0 || previous_frame_set.is_empty() || frame_set.len() < 2 {
            return None;
        }

        let pre_start = previous_frame_set.len().saturating_sub(frames);
        let post_count = frames.min(frame_set.len().saturating_sub(1));
        let mut best = f64::MAX;
        for pre_frame in &previous_frame_set[pre_start..] {
            for post_frame in frame_set.iter().skip(1).take(post_count) {
                let delta = self.masked_luma_delta_8bit(pre_frame, post_frame, mask_percent, 1, 1);
                if delta < best {
                    best = delta;
                }
            }
        }

        best.is_finite().then_some(best)
    }

    fn transient_similarity_threshold(&self, avg_luma_8bit: f64) -> f64 {
        let TransientSimilarityOptions {
            threshold_8bit,
            dark_threshold_8bit,
            dark_luma_low_8bit,
            dark_luma_high_8bit,
            ..
        } = self.tuning.transient_similarity;
        let darkness = smoothstep_darkness(avg_luma_8bit, dark_luma_low_8bit, dark_luma_high_8bit);
        threshold_8bit + (dark_threshold_8bit - threshold_8bit) * darkness
    }

    fn frame_similarity_lower_bound_8bit(
        &self,
        frame1: &Arc<Frame<T>>,
        frame2: &Arc<Frame<T>>,
        options: ForwardSimilarityOptions,
    ) -> f64 {
        let luma_delta = self.luma_delta_8bit(frame1, frame2);
        let luma_lower_bound = masked_delta_lower_bound_8bit(luma_delta, options.mask_percent);
        luma_lower_bound + self.weighted_chroma_delta_8bit(frame1, frame2, options.chroma_weight)
    }

    fn frame_similarity_delta_8bit(
        &mut self,
        frame1: &Arc<Frame<T>>,
        frame2: &Arc<Frame<T>>,
        options: ForwardSimilarityOptions,
    ) -> f64 {
        let luma_delta = if options.mask_percent > 0.0 {
            self.masked_luma_delta_8bit(
                frame1,
                frame2,
                options.mask_percent,
                options.mask_region_cols,
                options.mask_region_rows,
            )
        } else {
            self.luma_delta_8bit(frame1, frame2)
        };
        luma_delta + self.weighted_chroma_delta_8bit(frame1, frame2, options.chroma_weight)
    }

    fn weighted_chroma_delta_8bit(
        &self,
        frame1: &Arc<Frame<T>>,
        frame2: &Arc<Frame<T>>,
        chroma_weight: f64,
    ) -> f64 {
        let chroma_weight = chroma_weight.max(0.0);
        if chroma_weight == 0.0 {
            0.0
        } else {
            self.chroma_delta_8bit(frame1, frame2) * chroma_weight
        }
    }

    fn chroma_delta_8bit(&self, frame1: &Arc<Frame<T>>, frame2: &Arc<Frame<T>>) -> f64 {
        let mut total = 0.0;
        let mut planes = 0usize;
        if let (Some(plane1), Some(plane2)) = (&frame1.u_plane, &frame2.u_plane)
            && let Some(delta) = self.plane_delta_8bit(plane1, plane2)
        {
            total += delta;
            planes += 1;
        }
        if let (Some(plane1), Some(plane2)) = (&frame1.v_plane, &frame2.v_plane)
            && let Some(delta) = self.plane_delta_8bit(plane1, plane2)
        {
            total += delta;
            planes += 1;
        }

        if planes == 0 {
            0.0
        } else {
            total / planes as f64
        }
    }

    fn plane_delta_8bit(&self, plane1: &Plane<T>, plane2: &Plane<T>) -> Option<f64> {
        let width = plane1.width().get().min(plane2.width().get());
        let height = plane1.height().get().min(plane2.height().get());
        let pixels = width * height;
        if pixels == 0 {
            return None;
        }
        let scale = sample_range_scale(self.bit_depth);
        if plane1.width() == plane2.width() && plane1.height() == plane2.height() {
            return Some(sad_plane(plane1, plane2) as f64 / pixels as f64 / scale);
        }

        let stride1 = plane1.geometry().stride.get();
        let stride2 = plane2.geometry().stride.get();
        let origin1 = plane1.data_origin();
        let origin2 = plane2.data_origin();
        let data1 = plane1.data();
        let data2 = plane2.data();
        let mut total = 0.0;
        for y in 0..height {
            for x in 0..width {
                let idx1 = origin1 + y * stride1 + x;
                let idx2 = origin2 + y * stride2 + x;
                let p1 = data1[idx1].to_u16().expect("pixel value should fit in u16");
                let p2 = data2[idx2].to_u16().expect("pixel value should fit in u16");
                total += (p1 as f64 - p2 as f64).abs();
            }
        }
        Some(total / pixels as f64 / scale)
    }

    fn masked_luma_delta_8bit(
        &mut self,
        frame1: &Arc<Frame<T>>,
        frame2: &Arc<Frame<T>>,
        mask_percent: f64,
        mask_region_cols: usize,
        mask_region_rows: usize,
    ) -> f64 {
        const BLOCK_SIZE: usize = 32;
        const SAMPLE_STEP: usize = 4;

        let plane1 = &frame1.y_plane;
        let plane2 = &frame2.y_plane;
        let width = plane1.width().get().min(plane2.width().get());
        let height = plane1.height().get().min(plane2.height().get());
        let cols = width / BLOCK_SIZE;
        let rows = height / BLOCK_SIZE;
        if cols == 0 || rows == 0 {
            return self.luma_delta_8bit(frame1, frame2);
        }

        let stride1 = plane1.geometry().stride.get();
        let stride2 = plane2.geometry().stride.get();
        let origin1 = plane1.data_origin();
        let origin2 = plane2.data_origin();
        let data1 = plane1.data();
        let data2 = plane2.data();
        let scale = sample_range_scale(self.bit_depth);
        self.similarity_block_deltas.clear();
        self.similarity_block_deltas.reserve(cols * rows);

        for by in 0..rows {
            for bx in 0..cols {
                let mut total = 0.0;
                let mut count = 0usize;
                let y_base = by * BLOCK_SIZE;
                let x_base = bx * BLOCK_SIZE;
                for y in (y_base..y_base + BLOCK_SIZE).step_by(SAMPLE_STEP) {
                    for x in (x_base..x_base + BLOCK_SIZE).step_by(SAMPLE_STEP) {
                        let idx1 = origin1 + y * stride1 + x;
                        let idx2 = origin2 + y * stride2 + x;
                        let p1 = data1[idx1].to_u16().expect("pixel value should fit in u16");
                        let p2 = data2[idx2].to_u16().expect("pixel value should fit in u16");
                        total += (p1 as f64 - p2 as f64).abs();
                        count += 1;
                    }
                }
                self.similarity_block_deltas
                    .push(total / count as f64 / scale);
            }
        }

        self.masked_block_delta_mean(cols, rows, mask_percent, mask_region_cols, mask_region_rows)
    }

    fn masked_block_delta_mean(
        &mut self,
        cols: usize,
        rows: usize,
        mask_percent: f64,
        mask_region_cols: usize,
        mask_region_rows: usize,
    ) -> f64 {
        let block_count = self.similarity_block_deltas.len();
        if block_count == 0 {
            return 0.0;
        }

        let mask_percent = mask_percent.clamp(0.0, 0.95);
        let keep = ((block_count as f64 * (1.0 - mask_percent)).ceil() as usize)
            .max(1)
            .min(block_count);
        if keep == block_count {
            return self.similarity_block_deltas.iter().sum::<f64>() / block_count as f64;
        }

        if mask_region_cols <= 1 && mask_region_rows <= 1 {
            self.similarity_block_deltas
                .select_nth_unstable_by(keep - 1, |a, b| {
                    a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal)
                });
            return self.similarity_block_deltas[..keep].iter().sum::<f64>() / keep as f64;
        }

        self.spatially_capped_masked_block_delta_mean(
            cols,
            rows,
            mask_percent,
            mask_region_cols,
            mask_region_rows,
        )
    }

    fn spatially_capped_masked_block_delta_mean(
        &mut self,
        cols: usize,
        rows: usize,
        mask_percent: f64,
        mask_region_cols: usize,
        mask_region_rows: usize,
    ) -> f64 {
        let block_count = self.similarity_block_deltas.len();
        let target_mask = ((block_count as f64 * mask_percent).ceil() as usize)
            .min(block_count.saturating_sub(1));
        if target_mask == 0 {
            return self.similarity_block_deltas.iter().sum::<f64>() / block_count as f64;
        }

        let region_cols = mask_region_cols.clamp(1, cols);
        let region_rows = mask_region_rows.clamp(1, rows);
        let region_count = region_cols * region_rows;
        let per_region_cap = target_mask.div_ceil(region_count).max(1);

        self.similarity_block_mask.clear();
        self.similarity_block_mask.resize(block_count, false);
        self.similarity_block_indices.clear();
        self.similarity_block_indices.extend(0..block_count);
        self.similarity_region_counts.clear();
        self.similarity_region_counts.resize(region_count, 0);

        let block_deltas = &self.similarity_block_deltas;
        self.similarity_block_indices.sort_unstable_by(|&a, &b| {
            block_deltas[b]
                .partial_cmp(&block_deltas[a])
                .unwrap_or(cmp::Ordering::Equal)
        });

        let mut masked = 0usize;
        for index_idx in 0..self.similarity_block_indices.len() {
            if masked == target_mask {
                break;
            }
            let block_idx = self.similarity_block_indices[index_idx];
            let region_idx = block_region_index(block_idx, cols, rows, region_cols, region_rows);
            if self.similarity_region_counts[region_idx] >= per_region_cap {
                continue;
            }
            self.similarity_block_mask[block_idx] = true;
            self.similarity_region_counts[region_idx] += 1;
            masked += 1;
        }

        let mut total = 0.0;
        let mut kept = 0usize;
        for (idx, &delta) in self.similarity_block_deltas.iter().enumerate() {
            if !self.similarity_block_mask[idx] {
                total += delta;
                kept += 1;
            }
        }
        if kept == 0 {
            self.similarity_block_deltas.iter().sum::<f64>() / block_count as f64
        } else {
            total / kept as f64
        }
    }

    fn luma_delta_8bit(&self, frame1: &Arc<Frame<T>>, frame2: &Arc<Frame<T>>) -> f64 {
        let pixels = frame1.y_plane.width().get() * frame1.y_plane.height().get();
        if pixels == 0 {
            return 0.0;
        }
        let raw_delta = sad_plane(&frame1.y_plane, &frame2.y_plane) as f64 / pixels as f64;
        raw_delta / sample_range_scale(self.bit_depth)
    }
}

pub(crate) fn forward_similarity_start_allowed(
    options: ForwardSimilarityOptions,
    score: ScenecutResult,
    previous_scene_len: usize,
) -> bool {
    previous_scene_len >= options.min_previous_scene_len
        && score.cost_ratio >= options.min_cut_cost_ratio
}

pub(crate) fn forward_similarity_threshold_8bit(
    options: ForwardSimilarityOptions,
    score: ScenecutResult,
) -> f64 {
    let strict = options.threshold_8bit;
    let relaxed = options.relaxed_threshold_8bit;
    if relaxed <= strict {
        return strict;
    }

    let hard_cut =
        options.relaxed_min_cost_ratio > 0.0 && score.cost_ratio >= options.relaxed_min_cost_ratio;
    let strong_importance_cut = options.relaxed_importance_min_cost_ratio > 0.0
        && options.relaxed_min_imp_block_ratio > 0.0
        && score.cost_ratio >= options.relaxed_importance_min_cost_ratio
        && score.imp_block_ratio >= options.relaxed_min_imp_block_ratio;
    if hard_cut || strong_importance_cut {
        relaxed
    } else {
        strict
    }
}

fn sample_range_scale(bit_depth: usize) -> f64 {
    (2.0_f64.powi(bit_depth as i32) - 1.0) / 255.0
}

fn masked_delta_lower_bound_8bit(delta_8bit: f64, mask_percent: f64) -> f64 {
    let mask_percent = mask_percent.clamp(0.0, 0.95);
    if mask_percent == 0.0 {
        return delta_8bit;
    }

    ((delta_8bit - mask_percent * 255.0) / (1.0 - mask_percent)).max(0.0)
}

fn block_region_index(
    block_idx: usize,
    cols: usize,
    rows: usize,
    region_cols: usize,
    region_rows: usize,
) -> usize {
    debug_assert!(cols > 0);
    debug_assert!(rows > 0);
    debug_assert!(region_cols > 0);
    debug_assert!(region_rows > 0);

    let bx = block_idx % cols;
    let by = block_idx / cols;
    let region_x = (bx * region_cols / cols).min(region_cols - 1);
    let region_y = (by * region_rows / rows).min(region_rows - 1);
    region_y * region_cols + region_x
}

fn smoothstep_darkness(avg_luma_8bit: f64, low_luma_8bit: f64, high_luma_8bit: f64) -> f64 {
    if high_luma_8bit <= low_luma_8bit {
        return 0.0;
    }
    let t = ((high_luma_8bit - avg_luma_8bit) / (high_luma_8bit - low_luma_8bit)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score_with_ratios(cost_ratio: f64, imp_block_ratio: f64) -> ScenecutResult {
        let mut score = ScenecutResult::new(0.0, 0.0, 1.0, 1.0, 0.0);
        score.cost_ratio = cost_ratio;
        score.imp_block_ratio = imp_block_ratio;
        score
    }

    #[test]
    fn forward_similarity_start_gate_checks_previous_scene_and_cut_strength() {
        let options = ForwardSimilarityOptions {
            min_previous_scene_len: 18,
            min_cut_cost_ratio: 0.12,
            ..ForwardSimilarityOptions::default()
        };
        assert!(!forward_similarity_start_allowed(
            options,
            score_with_ratios(0.30, 3.0),
            17
        ));
        assert!(!forward_similarity_start_allowed(
            options,
            score_with_ratios(0.10, 3.0),
            30
        ));
        assert!(forward_similarity_start_allowed(
            options,
            score_with_ratios(0.30, 3.0),
            30
        ));
    }

    #[test]
    fn forward_similarity_relaxes_threshold_only_for_confident_cuts() {
        let options = ForwardSimilarityOptions {
            threshold_8bit: 6.0,
            relaxed_threshold_8bit: 7.5,
            relaxed_min_cost_ratio: 1.0,
            relaxed_importance_min_cost_ratio: 0.45,
            relaxed_min_imp_block_ratio: 3.6,
            ..ForwardSimilarityOptions::default()
        };
        assert_eq!(
            forward_similarity_threshold_8bit(options, score_with_ratios(0.43, 3.2)),
            6.0
        );
        assert_eq!(
            forward_similarity_threshold_8bit(options, score_with_ratios(1.05, 3.2)),
            7.5
        );
        assert_eq!(
            forward_similarity_threshold_8bit(options, score_with_ratios(0.49, 3.7)),
            7.5
        );
    }
}

fn record_forward_similarity_candidate(
    candidates: &mut [Option<ForwardSimilarityCandidate>; FORWARD_SIMILARITY_DIAGNOSTIC_SLOTS],
    candidate: ForwardSimilarityCandidate,
) {
    if let Some(slot) = candidates.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(candidate);
        sort_forward_similarity_candidates(candidates);
        return;
    }

    let Some((worst_idx, worst)) = candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, slot)| slot.map(|candidate| (idx, candidate)))
        .max_by(|(_, left), (_, right)| {
            left.delta
                .partial_cmp(&right.delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    else {
        return;
    };

    if candidate.decision == ForwardSimilarityCandidateDecision::Accepted
        || candidate.delta < worst.delta
    {
        candidates[worst_idx] = Some(candidate);
        sort_forward_similarity_candidates(candidates);
    }
}

fn forward_similarity_candidates_empty(
    candidates: &[Option<ForwardSimilarityCandidate>; FORWARD_SIMILARITY_DIAGNOSTIC_SLOTS],
) -> bool {
    candidates.iter().all(Option::is_none)
}

fn usize_is_zero(value: &usize) -> bool {
    *value == 0
}

fn sort_forward_similarity_candidates(
    candidates: &mut [Option<ForwardSimilarityCandidate>; FORWARD_SIMILARITY_DIAGNOSTIC_SLOTS],
) {
    candidates.sort_by(|left, right| match (left, right) {
        (Some(left), Some(right)) => left
            .delta
            .partial_cmp(&right.delta)
            .unwrap_or(std::cmp::Ordering::Equal),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
}

fn mark_top_blocks(source: &[f64], percent: f64, selected: &mut [bool]) {
    if source.is_empty() || selected.is_empty() || percent <= 0.0 {
        return;
    }

    let take = ((source.len() as f64 * percent).ceil() as usize)
        .max(1)
        .min(source.len());
    let mut indices = (0..source.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&a, &b| {
        source[b]
            .partial_cmp(&source[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for idx in indices.into_iter().take(take) {
        if let Some(slot) = selected.get_mut(idx) {
            *slot = true;
        }
    }
}

fn mark_top_blocks_spatially_capped(
    source: &[f64],
    percent: f64,
    block_cols: usize,
    block_rows: usize,
    region_cols: usize,
    region_rows: usize,
    selected: &mut [bool],
) {
    if source.is_empty()
        || selected.is_empty()
        || percent <= 0.0
        || block_cols == 0
        || block_rows == 0
        || region_cols == 0
        || region_rows == 0
    {
        return;
    }

    let target = ((source.len() as f64 * percent).ceil() as usize)
        .max(1)
        .min(source.len());
    let region_count = region_cols * region_rows;
    let per_region_cap = target.div_ceil(region_count).max(1);

    for region_y in 0..region_rows {
        let y_start = region_y * block_rows / region_rows;
        let y_end = ((region_y + 1) * block_rows / region_rows).min(block_rows);
        for region_x in 0..region_cols {
            let x_start = region_x * block_cols / region_cols;
            let x_end = ((region_x + 1) * block_cols / region_cols).min(block_cols);
            let mut indices = Vec::new();
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = y * block_cols + x;
                    if idx < source.len() {
                        indices.push(idx);
                    }
                }
            }

            indices.sort_unstable_by(|&a, &b| {
                source[b]
                    .partial_cmp(&source[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for idx in indices.into_iter().take(per_region_cap) {
                if let Some(slot) = selected.get_mut(idx) {
                    *slot = true;
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum ScenecutDecision {
    NotEvaluated,
    NoCut,
    Cut,
    CutImportance,
    SuppressedImportance,
    SuppressedFlash,
    SuppressedMinDistance,
    ForcedMaxDistance,
    SuppressedForwardSimilarity,
    SuppressedTransientSimilarity,
}

/// Contains the scores for scenecut analysis on a single frame
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct ScenecutResult {
    pub inter_cost: f64,
    pub imp_block_cost_raw: f64,
    pub imp_block_cost: f64,
    pub global_imp_block_cost: f64,
    pub imp_block_threshold: f64,
    pub imp_block_ratio: f64,
    pub global_imp_block_ratio: f64,
    pub backward_adjusted_cost: f64,
    pub forward_adjusted_cost: f64,
    pub threshold: f64,
    pub cost_ratio: f64,
    pub avg_luma_8bit: f64,
    pub me_bad_block_ratio: f64,
    pub me_good_block_ratio: f64,
    pub transient_similarity_score: Option<f64>,
    pub forward_similarity_score: Option<f64>,
    #[cfg_attr(
        feature = "serialize",
        serde(default, skip_serializing_if = "forward_similarity_candidates_empty")
    )]
    pub forward_similarity_candidates:
        [Option<ForwardSimilarityCandidate>; FORWARD_SIMILARITY_DIAGNOSTIC_SLOTS],
    #[cfg_attr(
        feature = "serialize",
        serde(default, skip_serializing_if = "usize_is_zero")
    )]
    pub forward_similarity_rejected_candidates: usize,
    pub decision: ScenecutDecision,
    pub forward_return_frame: Option<usize>,
}

impl ScenecutResult {
    #[inline]
    pub(crate) fn new(
        inter_cost: f64,
        imp_block_cost: f64,
        imp_block_threshold: f64,
        threshold: f64,
        avg_luma_8bit: f64,
    ) -> Self {
        let mut result = Self {
            inter_cost,
            imp_block_cost_raw: imp_block_cost,
            imp_block_cost,
            global_imp_block_cost: imp_block_cost,
            imp_block_threshold,
            imp_block_ratio: 0.0,
            global_imp_block_ratio: 0.0,
            backward_adjusted_cost: 0.0,
            forward_adjusted_cost: 0.0,
            threshold,
            cost_ratio: 0.0,
            avg_luma_8bit,
            me_bad_block_ratio: 0.0,
            me_good_block_ratio: 0.0,
            transient_similarity_score: None,
            forward_similarity_score: None,
            forward_similarity_candidates: [None; FORWARD_SIMILARITY_DIAGNOSTIC_SLOTS],
            forward_similarity_rejected_candidates: 0,
            decision: ScenecutDecision::NotEvaluated,
            forward_return_frame: None,
        };
        result.refresh_ratios();
        result
    }

    #[inline]
    pub(crate) fn refresh_ratios(&mut self) {
        self.cost_ratio = if self.threshold > 0.0 {
            self.forward_adjusted_cost / self.threshold
        } else {
            0.0
        };
        self.imp_block_ratio = if self.imp_block_threshold > 0.0 {
            self.imp_block_cost / self.imp_block_threshold
        } else {
            0.0
        };
        self.global_imp_block_ratio = if self.imp_block_threshold > 0.0 {
            self.global_imp_block_cost / self.imp_block_threshold
        } else {
            0.0
        };
    }
}
