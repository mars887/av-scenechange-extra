use std::{
    cmp,
    collections::BTreeMap,
    num::{NonZeroU8, NonZeroUsize},
    sync::Arc,
};

use log::debug;
use num_rational::Rational32;
use smallvec::SmallVec;
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
pub(crate) const FRAME_LUMA_SIGNATURE_CELLS: usize = 32;
const FRAME_LUMA_SIGNATURE_COLS: usize = 8;
const FRAME_LUMA_SIGNATURE_ROWS: usize = 4;
const FRAME_LUMA_SIGNATURE_SAMPLES_PER_CELL: usize = 4;
// Worst case across aggregations: 3 distinct percents x 2 mask kinds — the
// global `(1,1,false)` masks (feed `global_imp_block_cost`) plus the spatially
// capped `(region,region,true)` masks (feed `imp_block_cost`) — deduped by the
// cache. `high_quality` keeps 4 distinct masks (percents 0.10/0.15 x 2 kinds);
// sizing the inline buffer to 6 keeps every entry's mask set on the stack so the
// front-insert/back-pop deque churn never spills a `TopBlockMaskCache` to the heap.
const TOP_BLOCK_MASK_CACHE_INLINE_CAPACITY: usize = 6;
const FORWARD_REFERENCE_INLINE_CAPACITY: usize = 8;

// Forward-similarity block-sum prefilter (P4 / Variant D). The signature samples
// exactly the lattice that `masked_luma_delta_8bit` reads (`SAMPLE_STEP` inside
// full `BLOCK_SIZE`x`BLOCK_SIZE` blocks) but stores it as four 16x16 sub-area
// integer sums per block, so a triangle-inequality lower bound on each block's
// masked-luma delta can be assembled without re-reading any planes.
const FORWARD_PREFILTER_BLOCK_SIZE: usize = 32;
const FORWARD_PREFILTER_SAMPLE_STEP: usize = 4;
const FORWARD_PREFILTER_SUBAREA_SIZE: usize = 16;
const FORWARD_PREFILTER_SUBAREAS: usize = 4;
/// Number of sampled points per block (`(BLOCK_SIZE / SAMPLE_STEP)^2`), the
/// divisor `masked_luma_delta_8bit` uses for a full block.
const FORWARD_PREFILTER_SAMPLES_PER_BLOCK: f64 = 64.0;
/// Conservative slack subtracted before rejecting on the prefilter bound, so f64
/// accumulation rounding can never turn a real `bound <= threshold` into a reject
/// (the bound is a strict lower bound by construction; this only guards float
/// arithmetic and dwarfs the worst-case accumulation error).
const FORWARD_PREFILTER_ROUNDING_GUARD_8BIT: f64 = 1e-6;

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

/// Reader-side plan for shrinking the parallel frame store (P2/P3).
///
/// The only chroma consumer in the analysis path is forward-similarity, and the
/// only consumer of a box-downscaled luma plane is the Fast detector. So the
/// parallel reader may store a reduced frame whenever those paths are inactive:
///   * `drop_chroma` (forward-similarity disabled) drops both chroma planes;
///   * `prescale` (Fast, >240p, with forward- AND transient-similarity disabled)
///     replaces luma with its box-downscale so `fast_scenecut` skips its own.
///
/// When neither applies the frame is stored unchanged, preserving the
/// serial/parallel result parity the chunk handoff relies on. The gate keys off
/// the tuning flags (not the speed enum), because speed and tuning are
/// independent: e.g. `Standard` speed with `high_quality()` tuning still reads
/// chroma and must keep it.
pub(crate) struct ParallelFrameReducer<T: Pixel> {
    drop_chroma: bool,
    prescale: Option<ScaleFunction<T>>,
}

impl<T: Pixel> ParallelFrameReducer<T> {
    pub(crate) fn new(
        resolution: (usize, usize),
        speed: SceneDetectionSpeed,
        forward_similarity_enabled: bool,
        transient_similarity_enabled: bool,
    ) -> Self {
        // Pre-downscale only when nothing needs full-resolution luma. The Fast
        // detector is the sole downscaled-luma consumer; forward/transient
        // similarity read full-res luma (and forward additionally reads chroma).
        let prescale = if forward_similarity_enabled || transient_similarity_enabled {
            None
        } else {
            // `detect_scale_factor` is `None` unless `speed == Fast` and the
            // smaller edge is >240px, so Standard/High never pre-downscale.
            detect_scale_factor::<T>(resolution, speed)
        };
        Self {
            drop_chroma: !forward_similarity_enabled,
            prescale,
        }
    }

    /// Whether workers fed by this store receive already-downscaled luma.
    pub(crate) fn is_prescaled(&self) -> bool {
        self.prescale.is_some()
    }

    /// Whether chroma is dropped from the stored frames.
    pub(crate) fn drops_chroma(&self) -> bool {
        self.drop_chroma
    }

    /// The Fast downscale factor applied to stored luma, if any.
    pub(crate) fn scale_factor(&self) -> Option<usize> {
        self.prescale.as_ref().map(|scale| scale.factor.get())
    }

    /// Reduces a freshly decoded frame to the payload the store should hold.
    /// Consumes `frame`, moving the luma plane (never copying) when only chroma
    /// is dropped. Chroma is read only by `forward_similarity`, so a luma-only
    /// frame is parity-safe whenever `drop_chroma` is set.
    pub(crate) fn apply(&self, frame: Frame<T>) -> Frame<T> {
        if let Some(scale) = &self.prescale {
            let bit_depth = frame.bit_depth;
            return Frame {
                y_plane: (scale.downscale)(&frame.y_plane, bit_depth),
                u_plane: None,
                v_plane: None,
                subsampling: ChromaSubsampling::Monochrome,
                bit_depth,
            };
        }
        if self.drop_chroma {
            let Frame {
                y_plane, bit_depth, ..
            } = frame;
            return Frame {
                y_plane,
                u_plane: None,
                v_plane: None,
                subsampling: ChromaSubsampling::Monochrome,
                bit_depth,
            };
        }
        frame
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

#[derive(Clone, Copy, Debug)]
struct ForwardReturnCandidate {
    offset: usize,
    score: ScenecutResult,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum ForwardSimilarityCandidateDecision {
    Accepted,
    MissingReturnCandidate,
    DominantReturnCandidate,
    DeltaAboveThreshold,
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
struct TopBlockMaskCache {
    percent_bits: u64,
    region_cols: usize,
    region_rows: usize,
    spatially_capped: bool,
    selected: Vec<bool>,
}

#[derive(Clone, Debug)]
pub(crate) struct ScenecutAnalysis {
    pub result: ScenecutResult,
    pub importance_blocks: Vec<f64>,
    pub importance_cols: usize,
    pub importance_rows: usize,
    top_block_masks: SmallVec<[TopBlockMaskCache; TOP_BLOCK_MASK_CACHE_INLINE_CAPACITY]>,
    /// P5 memo: the `(previous_present, next_present)` neighbour shape this entry
    /// last refreshed against. Its importance scores depend only on the (frozen
    /// after construction) importance blocks of itself and its present
    /// previous/next deque neighbours, and the deque is only ever front-inserted
    /// / back-popped — so an unchanged shape means an unchanged score and
    /// `refresh_importance_metrics` can skip the recompute. `None` until first
    /// refresh; `Mean` keeps it `(false, false)` (computed once, then skipped).
    importance_neighbor_state: Option<(bool, bool)>,
}

impl ScenecutAnalysis {
    #[inline]
    fn from_result(result: ScenecutResult) -> Self {
        Self {
            result,
            importance_blocks: Vec::new(),
            importance_cols: 0,
            importance_rows: 0,
            top_block_masks: SmallVec::new(),
            importance_neighbor_state: None,
        }
    }

    fn cache_top_block_mask(
        &mut self,
        percent: f64,
        region_cols: usize,
        region_rows: usize,
        spatially_capped: bool,
    ) {
        if self.importance_blocks.is_empty() || percent <= 0.0 {
            return;
        }
        let percent_bits = percent.to_bits();
        if self.top_block_masks.iter().any(|entry| {
            entry.percent_bits == percent_bits
                && entry.region_cols == region_cols
                && entry.region_rows == region_rows
                && entry.spatially_capped == spatially_capped
        }) {
            return;
        }

        let mut selected = vec![false; self.importance_blocks.len()];
        if spatially_capped {
            mark_top_blocks_spatially_capped(
                &self.importance_blocks,
                percent,
                self.importance_cols,
                self.importance_rows,
                region_cols,
                region_rows,
                &mut selected,
            );
        } else {
            mark_top_blocks(&self.importance_blocks, percent, &mut selected);
        }
        self.top_block_masks.push(TopBlockMaskCache {
            percent_bits,
            region_cols,
            region_rows,
            spatially_capped,
            selected,
        });
    }

    fn cached_top_block_mask(
        &self,
        percent: f64,
        region_cols: usize,
        region_rows: usize,
        spatially_capped: bool,
    ) -> Option<&[bool]> {
        let percent_bits = percent.to_bits();
        self.top_block_masks
            .iter()
            .find(|entry| {
                entry.percent_bits == percent_bits
                    && entry.region_cols == region_cols
                    && entry.region_rows == region_rows
                    && entry.spatially_capped == spatially_capped
            })
            .map(|entry| entry.selected.as_slice())
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
    /// Set on parallel Fast workers whose store frames already carry box-
    /// downscaled luma (P2). When `true`, `fast_scenecut` SADs the supplied luma
    /// planes directly instead of downscaling again; `scaled_pixels` already
    /// holds the downscaled count, so the result is bit-identical. The serial
    /// path feeds full frames and leaves this `false`.
    frames_pre_downscaled: bool,

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
    /// Per-search block-sum signatures of the forward-similarity reference window
    /// (Variant D prefilter); rebuilt once at the start of each search.
    forward_ref_signatures: Vec<ForwardPrefilterSignature>,
    /// Per-search lazily-built block-sum signatures of the forward `frame_set`,
    /// indexed by `frame_set` position so overlapping offsets reuse them.
    forward_post_signatures: Vec<Option<ForwardPrefilterSignature>>,
    /// Reused per-block triangle lower bounds for the Variant D prefilter.
    prefilter_block_bounds: Vec<f64>,
    /// Reused mask for spatially capped volatile blocks.
    similarity_block_mask: Vec<bool>,
    /// Reused block indices for spatially capped volatile-block selection.
    similarity_block_indices: Vec<usize>,
    /// Reused per-region masked block counts.
    similarity_region_counts: Vec<usize>,
    /// Reused `selected` scratch for the temporal top-block scorers (P5), so
    /// `refresh_importance_metrics` stops allocating `vec![false; block_count]`
    /// per call. Length-stable across a run (every entry has the same block
    /// count); detached via `mem::take` while scoring, then restored.
    importance_selected_scratch: Vec<bool>,
    /// Temporary buffer used by `estimate_intra_costs`.
    /// We store it on the struct so we only need to allocate it once.
    temp_plane: Option<Plane<T>>,
    /// Buffer for `FrameMEStats` for cost scenecut
    frame_me_stats_buffer: Option<RefMEStats>,
    /// Whether a single cost comparison may split intra/inter/importance into
    /// rayon jobs.
    use_cost_parallelism: bool,

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
            forward_ref_signatures: Vec::new(),
            forward_post_signatures: Vec::new(),
            prefilter_block_bounds: Vec::new(),
            similarity_block_mask: Vec::new(),
            similarity_block_indices: Vec::new(),
            similarity_region_counts: Vec::new(),
            importance_selected_scratch: Vec::new(),
            scaled_pixels: pixels,
            bit_depth,
            frame_rate,
            chroma_sampling,
            min_key_frame_interval,
            max_key_frame_interval,
            downscaled_frame_buffer: None,
            frames_pre_downscaled: false,
            resolution,
            temp_plane: None,
            frame_me_stats_buffer: None,
            use_cost_parallelism: true,
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

    pub(crate) fn set_cost_parallelism(&mut self, enabled: bool) {
        self.use_cost_parallelism = enabled;
    }

    /// Marks that stored frames already carry box-downscaled luma (P2 parallel
    /// reduced store), so `fast_scenecut` must not downscale them again. Only the
    /// parallel Fast workers set this; the serial path leaves it `false`.
    pub(crate) fn set_frames_pre_downscaled(&mut self, enabled: bool) {
        self.frames_pre_downscaled = enabled;
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
                analysis.result.motion_backward_adjusted_cost = 0.0;
            } else {
                analysis.result.backward_adjusted_cost = adjusted_peak_cost(
                    analysis.result.inter_cost,
                    self.score_deque
                        .iter()
                        .take(self.deque_offset)
                        .map(|i| i.result.inter_cost),
                );
                analysis.result.motion_backward_adjusted_cost = adjusted_peak_cost(
                    analysis.result.motion_inter_cost,
                    self.score_deque
                        .iter()
                        .take(self.deque_offset)
                        .map(|i| i.result.motion_inter_cost),
                );
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
                    let motion_adjusted_cost = self.score_deque[i].result.motion_inter_cost
                        - analysis.result.motion_inter_cost;
                    if i == 0
                        || motion_adjusted_cost
                            < self.score_deque[i].result.motion_forward_adjusted_cost
                    {
                        self.score_deque[i].result.motion_forward_adjusted_cost =
                            motion_adjusted_cost;
                    }
                    if self.score_deque[i].result.motion_forward_adjusted_cost < 0.0 {
                        self.score_deque[i].result.motion_forward_adjusted_cost = 0.0;
                    }
                    self.score_deque[i].result.refresh_ratios();
                }
            }
        }
        self.prepare_importance_top_block_masks(&mut analysis);
        self.score_deque.insert(0, analysis);
    }

    fn prepare_importance_top_block_masks(&self, analysis: &mut ScenecutAnalysis) {
        match self.tuning.importance_aggregation {
            ImportanceAggregation::Mean => {}
            ImportanceAggregation::TemporalTopBlocks {
                previous_percent,
                current_percent,
                next_percent,
            } => {
                analysis.cache_top_block_mask(previous_percent, 1, 1, false);
                analysis.cache_top_block_mask(current_percent, 1, 1, false);
                analysis.cache_top_block_mask(next_percent, 1, 1, false);
            }
            ImportanceAggregation::SpatialTemporalTopBlocks {
                previous_percent,
                current_percent,
                next_percent,
                region_cols,
                region_rows,
            } => {
                // Cache exactly the masks the two readers in
                // `refresh_importance_metrics` look up. The spatially capped
                // `(region,region,true)` masks feed `imp_block_cost`
                // (`temporal_top_importance_score_spatially_capped`); the
                // `(1,1,false)` masks feed `global_imp_block_cost`
                // (`temporal_top_importance_score`). Caching both at insertion (P5)
                // makes the per-frame refresh pure cached-scan + average, so it
                // never re-sorts the 32k-block index vector on either path. Keep
                // these two lists in lockstep with those lookups.
                analysis.cache_top_block_mask(previous_percent, region_cols, region_rows, true);
                analysis.cache_top_block_mask(current_percent, region_cols, region_rows, true);
                analysis.cache_top_block_mask(next_percent, region_cols, region_rows, true);
                analysis.cache_top_block_mask(previous_percent, 1, 1, false);
                analysis.cache_top_block_mask(current_percent, 1, 1, false);
                analysis.cache_top_block_mask(next_percent, 1, 1, false);
            }
        }
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
        if self.tuning.importance_cut_max_me_good_ratio > 0.0
            && current.static_good_block_ratio > self.tuning.importance_cut_max_me_good_ratio
        {
            return false;
        }
        if self.tuning.importance_cut_bright_max_cost_ratio > 0.0
            && current.avg_luma_8bit > self.tuning.importance_cut_dark_luma_high_8bit
            && current.cost_ratio > self.tuning.importance_cut_bright_max_cost_ratio
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
                    && current.static_bad_block_ratio >= self.tuning.importance_cut_min_me_bad_ratio
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
        // P5: an entry's importance scores read only its own (frozen) importance
        // blocks plus those of its present previous/next neighbours, so they can
        // change only when a neighbour appears or disappears. Skip the recompute
        // (and its sorts/allocations) when the neighbour shape is unchanged since
        // the last refresh. Keying inside this method — rather than narrowing the
        // caller's `0..len` loop — keeps the skip valid for any future deque
        // maintenance, because it observes the real inputs. `Mean` does not vary
        // with neighbours, so it collapses to `(false, false)`: computed once,
        // then skipped.
        let is_temporal = matches!(
            self.tuning.importance_aggregation,
            ImportanceAggregation::TemporalTopBlocks { .. }
                | ImportanceAggregation::SpatialTemporalTopBlocks { .. }
        );
        let neighbor_state = (
            is_temporal && index + 1 < self.score_deque.len(),
            is_temporal && index > 0,
        );
        if analysis.importance_neighbor_state == Some(neighbor_state) {
            return;
        }
        let raw = analysis.result.imp_block_cost_raw;
        let avg_luma_8bit = analysis.result.avg_luma_8bit;

        // Detach the shared `selected` scratch so the `&self` scorers can reuse
        // its allocation across frames instead of a per-call `vec![false; n]`.
        // No early return runs between here and the restore below.
        let mut selected_scratch = std::mem::take(&mut self.importance_selected_scratch);
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
                    &mut selected_scratch,
                    index,
                    previous_percent,
                    current_percent,
                    next_percent,
                )
                .unwrap_or(raw),
            ImportanceAggregation::Mean => raw,
        };
        let imp_block_cost = match self.tuning.importance_aggregation {
            ImportanceAggregation::Mean => raw,
            ImportanceAggregation::TemporalTopBlocks {
                previous_percent,
                current_percent,
                next_percent,
            } => self
                .temporal_top_importance_score(
                    &mut selected_scratch,
                    index,
                    previous_percent,
                    current_percent,
                    next_percent,
                )
                .unwrap_or(raw),
            ImportanceAggregation::SpatialTemporalTopBlocks {
                previous_percent,
                current_percent,
                next_percent,
                region_cols,
                region_rows,
            } => self
                .temporal_top_importance_score_spatially_capped(
                    &mut selected_scratch,
                    index,
                    previous_percent,
                    current_percent,
                    next_percent,
                    region_cols,
                    region_rows,
                )
                .unwrap_or(raw),
        };
        self.importance_selected_scratch = selected_scratch;
        let threshold = self.importance_threshold(avg_luma_8bit);
        if let Some(analysis) = self.score_deque.get_mut(index) {
            analysis.result.imp_block_cost = imp_block_cost;
            analysis.result.global_imp_block_cost = global_imp_block_cost;
            analysis.result.imp_block_threshold = threshold;
            analysis.result.refresh_ratios();
            analysis.importance_neighbor_state = Some(neighbor_state);
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
        selected_buf: &mut Vec<bool>,
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

        selected_buf.clear();
        selected_buf.resize(block_count, false);
        let selected = selected_buf.as_mut_slice();
        if let Some(previous) = self.score_deque.get(index + 1) {
            if let Some(mask) = previous.cached_top_block_mask(previous_percent, 1, 1, false) {
                mark_cached_top_blocks(mask, selected);
            } else {
                mark_top_blocks(&previous.importance_blocks, previous_percent, selected);
            }
        }
        if let Some(mask) = current.cached_top_block_mask(current_percent, 1, 1, false) {
            mark_cached_top_blocks(mask, selected);
        } else {
            mark_top_blocks(&current.importance_blocks, current_percent, selected);
        }
        if index > 0
            && let Some(next) = self.score_deque.get(index - 1)
        {
            if let Some(mask) = next.cached_top_block_mask(next_percent, 1, 1, false) {
                mark_cached_top_blocks(mask, selected);
            } else {
                mark_top_blocks(&next.importance_blocks, next_percent, selected);
            }
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

    #[allow(clippy::too_many_arguments)]
    fn temporal_top_importance_score_spatially_capped(
        &self,
        selected_buf: &mut Vec<bool>,
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

        selected_buf.clear();
        selected_buf.resize(block_count, false);
        let selected = selected_buf.as_mut_slice();
        if let Some(previous) = self.score_deque.get(index + 1) {
            if let Some(mask) =
                previous.cached_top_block_mask(previous_percent, region_cols, region_rows, true)
            {
                mark_cached_top_blocks(mask, selected);
            } else {
                mark_top_blocks_spatially_capped(
                    &previous.importance_blocks,
                    previous_percent,
                    current.importance_cols,
                    current.importance_rows,
                    region_cols,
                    region_rows,
                    selected,
                );
            }
        }
        if let Some(mask) =
            current.cached_top_block_mask(current_percent, region_cols, region_rows, true)
        {
            mark_cached_top_blocks(mask, selected);
        } else {
            mark_top_blocks_spatially_capped(
                &current.importance_blocks,
                current_percent,
                current.importance_cols,
                current.importance_rows,
                region_cols,
                region_rows,
                selected,
            );
        }
        if index > 0
            && let Some(next) = self.score_deque.get(index - 1)
        {
            if let Some(mask) =
                next.cached_top_block_mask(next_percent, region_cols, region_rows, true)
            {
                mark_cached_top_blocks(mask, selected);
            } else {
                mark_top_blocks_spatially_capped(
                    &next.importance_blocks,
                    next_percent,
                    current.importance_cols,
                    current.importance_rows,
                    region_cols,
                    region_rows,
                    selected,
                );
            }
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
            .collect::<SmallVec<[&Arc<Frame<T>>; FORWARD_REFERENCE_INLINE_CAPACITY]>>();
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
        // A segment delta may only be reported as a non-exact luma lower bound
        // strictly above this; at or below it the exact masked-luma+chroma value
        // is returned so every downstream `candidate.delta <= K` consumer sees the
        // exact value (effective forward threshold or the option-independent
        // text-card/dark-occlusion auxiliaries, whichever is larger).
        let confirmation_threshold =
            threshold_8bit.max(crate::FORWARD_SIMILARITY_AUXILIARY_MAX_THRESHOLD_8BIT);

        // Variant D prefilter: build the reference-window block-sum signatures once
        // for this search and reset the lazily-filled post-frame cache. Only when
        // masking is active (the signatures represent the masked-luma lattice).
        self.forward_ref_signatures.clear();
        self.forward_post_signatures.clear();
        if options.mask_percent > 0.0 {
            for frame in reference_window.iter().copied() {
                self.forward_ref_signatures
                    .push(build_forward_prefilter_signature(frame));
            }
            self.forward_post_signatures
                .resize_with(frame_set.len(), || None);
        }

        let mut candidates = [None; FORWARD_SIMILARITY_DIAGNOSTIC_SLOTS];
        let mut accepted = None;
        let mut rejected_candidates = 0usize;
        for offset in min_offset..=max_offset {
            let return_candidate = self.forward_return_candidate(min_offset, offset);
            let return_candidate_offset =
                return_candidate.map(|return_candidate| return_candidate.offset);

            let Some(post_start_offset) = self.forward_similarity_post_start_offset(
                offset,
                return_candidate_offset,
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
                confirmation_threshold,
            );
            let allow_flash_return = require_return_candidate
                && return_candidate.is_none()
                && forward_similarity_flash_return_without_candidate_allowed(
                    options, score, offset, delta,
                );
            let decision =
                if require_return_candidate && return_candidate.is_none() && !allow_flash_return {
                    ForwardSimilarityCandidateDecision::MissingReturnCandidate
                } else if let Some(return_candidate) = return_candidate
                    && !forward_similarity_return_candidate_allowed(
                        options,
                        score,
                        return_candidate.score,
                        delta,
                    )
                {
                    ForwardSimilarityCandidateDecision::DominantReturnCandidate
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
                candidate_frame: return_candidate_offset
                    .map(|candidate_offset| input_frameno + candidate_offset - 1),
                candidate_offset: return_candidate_offset,
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
        confirmation_threshold: f64,
    ) -> f64 {
        let comparison_frames = reference_window
            .len()
            .min(post_end_offset + 1 - post_start_offset);
        if comparison_frames == 0 {
            return f64::MAX;
        }

        let reference_start = reference_window.len() - comparison_frames;

        // Stage 0 — cached block-sum prefilter (P4 / Variant D). When masking is
        // active and signatures were built for this search, assemble a strict
        // lower bound on the masked-luma metric from the cached per-frame block-sum
        // signatures (no full-plane reads, and post frames are sampled once and
        // reused across overlapping offsets). If even that bound clears the
        // confirmation threshold the exact metric does too, so report the bound and
        // skip every plane read. The rounding guard keeps a true `<=` from being
        // rejected by f64 accumulation error.
        if options.mask_percent > 0.0
            && !self.forward_ref_signatures.is_empty()
            && let Some(bound) = self.forward_segment_prefilter_bound(
                frame_set,
                post_start_offset,
                reference_start,
                comparison_frames,
                options.mask_percent,
            )
            && bound - FORWARD_PREFILTER_ROUNDING_GUARD_8BIT > confirmation_threshold
        {
            return bound;
        }

        // Stage 1 — exact final luma (P4 / Variant C). Compute the luma metric the
        // segment delta is actually built from (the masked 1/16-lattice metric, or
        // the full-res SAD when masking is off) and cache each per-frame value.
        // This replaces the former full-res luma SAD fed through
        // `masked_delta_lower_bound_8bit`: that analytic bound was derived from the
        // all-pixel SAD, but the refined metric only samples a 1/16 lattice over
        // full 32x32 blocks, so it was NOT a valid lower bound on the metric and
        // could reject pairs the exact metric accepts.
        let mut luma_deltas =
            SmallVec::<[f64; FORWARD_REFERENCE_INLINE_CAPACITY]>::with_capacity(comparison_frames);
        let mut luma_mean = 0.0;
        for idx in 0..comparison_frames {
            let frame1 = reference_window[reference_start + idx];
            let frame2 = frame_set[post_start_offset + idx];
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
            luma_deltas.push(luma_delta);
            luma_mean += luma_delta;
        }
        luma_mean /= comparison_frames as f64;

        // Weighted chroma is non-negative, so the luma mean is a strict lower bound
        // on the final segment delta. If it already exceeds every threshold that
        // can act on this candidate, the exact delta does too: skip both full-res
        // chroma plane SADs and report the luma mean (a value strictly above
        // `confirmation_threshold`, hence inert to every `delta <= K` consumer).
        if luma_mean > confirmation_threshold {
            return luma_mean;
        }

        // Stage 2 — luma survivors get the exact chroma term, computed once per
        // pair (not twice as before) and combined in frame order with the cached
        // luma delta, so the surviving (consumable) delta is the exact metric.
        let mut delta = 0.0;
        for idx in 0..comparison_frames {
            let frame1 = reference_window[reference_start + idx];
            let frame2 = frame_set[post_start_offset + idx];
            let weighted_chroma =
                self.weighted_chroma_delta_8bit(frame1, frame2, options.chroma_weight);
            delta += luma_deltas[idx] + weighted_chroma;
        }
        delta / comparison_frames as f64
    }

    /// Strict lower bound on the segment's masked-luma metric assembled from the
    /// cached block-sum signatures (Variant D), or `None` when any pair has no full
    /// block grid in common (the exact path then handles it). The bound is `<=` the
    /// exact masked-luma mean of every pair: each block's bound is a sub-area
    /// triangle-inequality lower bound on that block's sampled delta, and keeping
    /// the smallest `keep` of them is the minimum-sum subset of its size, so it
    /// bounds both the simple and spatially-capped exact masked means.
    fn forward_segment_prefilter_bound(
        &mut self,
        frame_set: &[&Arc<Frame<T>>],
        post_start_offset: usize,
        reference_start: usize,
        comparison_frames: usize,
        mask_percent: f64,
    ) -> Option<f64> {
        // Build any post-frame signatures not cached yet for this search.
        for idx in 0..comparison_frames {
            let post_index = post_start_offset + idx;
            if self.forward_post_signatures[post_index].is_none() {
                self.forward_post_signatures[post_index] =
                    Some(build_forward_prefilter_signature(frame_set[post_index]));
            }
        }

        let scale = sample_range_scale(self.bit_depth);
        let clamped = mask_percent.clamp(0.0, 0.95);
        let mut segment_bound = 0.0;
        for idx in 0..comparison_frames {
            let ref_sig = &self.forward_ref_signatures[reference_start + idx];
            let post_sig = self.forward_post_signatures[post_start_offset + idx]
                .as_ref()
                .expect("post signature built above");
            let cols = ref_sig.cols.min(post_sig.cols);
            let rows = ref_sig.rows.min(post_sig.rows);
            if cols == 0 || rows == 0 {
                return None;
            }
            let block_count = cols * rows;
            // Keep the same block count the exact masking retains (it removes
            // `ceil(n*mask)` blocks, clamped to keep >= 1); valid for both the
            // simple and spatially-capped paths.
            let target_mask =
                ((block_count as f64 * clamped).ceil() as usize).min(block_count - 1);
            let keep = block_count - target_mask;

            self.prefilter_block_bounds.clear();
            self.prefilter_block_bounds.reserve(block_count);
            for by in 0..rows {
                for bx in 0..cols {
                    let reference = ref_sig.sub_sums[by * ref_sig.cols + bx];
                    let post = post_sig.sub_sums[by * post_sig.cols + bx];
                    let mut numerator = 0u32;
                    for sub in 0..FORWARD_PREFILTER_SUBAREAS {
                        numerator += reference[sub].abs_diff(post[sub]);
                    }
                    self.prefilter_block_bounds
                        .push(f64::from(numerator) / FORWARD_PREFILTER_SAMPLES_PER_BLOCK / scale);
                }
            }
            segment_bound += forward_prefilter_masked_mean(&mut self.prefilter_block_bounds, keep);
        }

        Some(segment_bound / comparison_frames as f64)
    }

    fn forward_return_candidate(
        &self,
        min_offset: usize,
        return_offset: usize,
    ) -> Option<ForwardReturnCandidate> {
        (min_offset..=return_offset)
            .rev()
            .find_map(|candidate_offset| {
                self.forward_return_candidate_score(candidate_offset)
                    .map(|score| ForwardReturnCandidate {
                        offset: candidate_offset,
                        score,
                    })
            })
    }

    fn forward_return_candidate_score(&self, offset: usize) -> Option<ScenecutResult> {
        let frame_offset = offset.saturating_sub(1);
        if frame_offset == 0 || frame_offset > self.deque_offset {
            return None;
        }
        let index = self.deque_offset - frame_offset;
        let score = self.score_deque.get(index)?.result;
        (score.forward_adjusted_cost >= score.threshold || self.importance_cut_passed(index))
            .then_some(score)
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
    if options.relaxed_max_cost_ratio > 0.0 && score.cost_ratio > options.relaxed_max_cost_ratio {
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

pub(crate) fn forward_similarity_return_candidate_allowed(
    options: ForwardSimilarityOptions,
    score: ScenecutResult,
    return_candidate_score: ScenecutResult,
    delta: f64,
) -> bool {
    if delta <= options.threshold_8bit {
        return true;
    }
    if forward_similarity_threshold_8bit(options, score) <= options.threshold_8bit {
        return true;
    }

    let max_return = options.relaxed_max_return_cost_ratio;
    let max_multiplier = options.relaxed_max_return_cost_ratio_multiplier;
    if max_return <= 0.0 || max_multiplier <= 0.0 {
        return true;
    }

    return_candidate_score.cost_ratio <= max_return
        || return_candidate_score.cost_ratio <= score.cost_ratio * max_multiplier
}

fn forward_similarity_flash_return_without_candidate_allowed(
    options: ForwardSimilarityOptions,
    score: ScenecutResult,
    offset: usize,
    delta: f64,
) -> bool {
    options.flash_return_without_candidate
        && options.flash_return_frames > 0
        && offset <= options.flash_return_frames
        && score.cost_ratio >= options.flash_return_min_cost_ratio
        && delta <= options.threshold_8bit
}

fn sample_range_scale(bit_depth: usize) -> f64 {
    (2.0_f64.powi(bit_depth as i32) - 1.0) / 255.0
}

/// Block-sum signature backing the Variant D forward-similarity prefilter.
///
/// For every full `FORWARD_PREFILTER_BLOCK_SIZE` block of a frame's luma plane it
/// stores the raw integer sums of the sampled lattice points (`SAMPLE_STEP`) in
/// each of the four `FORWARD_PREFILTER_SUBAREA_SIZE` quadrants. The sums are
/// bit-depth independent; the sample-range scale is applied when the bound is
/// formed. `cols`/`rows` are this frame's own full-block grid, so a pair compares
/// only the common `min(cols)` x `min(rows)` sub-grid.
struct ForwardPrefilterSignature {
    cols: usize,
    rows: usize,
    sub_sums: Vec<[u32; FORWARD_PREFILTER_SUBAREAS]>,
}

fn build_forward_prefilter_signature<T: Pixel>(frame: &Frame<T>) -> ForwardPrefilterSignature {
    let plane = &frame.y_plane;
    let cols = plane.width().get() / FORWARD_PREFILTER_BLOCK_SIZE;
    let rows = plane.height().get() / FORWARD_PREFILTER_BLOCK_SIZE;
    let mut sub_sums = Vec::with_capacity(cols * rows);
    if cols == 0 || rows == 0 {
        return ForwardPrefilterSignature {
            cols,
            rows,
            sub_sums,
        };
    }

    let stride = plane.geometry().stride.get();
    let origin = plane.data_origin();
    let data = plane.data();
    for by in 0..rows {
        for bx in 0..cols {
            let y_base = by * FORWARD_PREFILTER_BLOCK_SIZE;
            let x_base = bx * FORWARD_PREFILTER_BLOCK_SIZE;
            let mut sums = [0u32; FORWARD_PREFILTER_SUBAREAS];
            let mut dy = 0;
            while dy < FORWARD_PREFILTER_BLOCK_SIZE {
                let quad_y = dy / FORWARD_PREFILTER_SUBAREA_SIZE;
                let mut dx = 0;
                while dx < FORWARD_PREFILTER_BLOCK_SIZE {
                    let quad = quad_y * 2 + dx / FORWARD_PREFILTER_SUBAREA_SIZE;
                    let pixel = data[origin + (y_base + dy) * stride + (x_base + dx)]
                        .to_u32()
                        .expect("pixel value should fit in u32");
                    sums[quad] += pixel;
                    dx += FORWARD_PREFILTER_SAMPLE_STEP;
                }
                dy += FORWARD_PREFILTER_SAMPLE_STEP;
            }
            sub_sums.push(sums);
        }
    }

    ForwardPrefilterSignature {
        cols,
        rows,
        sub_sums,
    }
}

/// Mean of the smallest `keep` block bounds — the global-masking lower bound that
/// is `<=` both the simple and spatially-capped exact masked means (the smallest
/// `keep` form the minimum-sum subset of their size). Partially reorders `bounds`.
fn forward_prefilter_masked_mean(bounds: &mut [f64], keep: usize) -> f64 {
    let block_count = bounds.len();
    if block_count == 0 {
        return 0.0;
    }
    let keep = keep.max(1).min(block_count);
    if keep == block_count {
        return bounds.iter().sum::<f64>() / block_count as f64;
    }
    bounds.select_nth_unstable_by(keep - 1, |a, b| {
        a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal)
    });
    bounds[..keep].iter().sum::<f64>() / keep as f64
}

pub(crate) fn frame_luma_signature_8bit<T: Pixel>(
    frame: &Frame<T>,
    bit_depth: usize,
) -> [u8; FRAME_LUMA_SIGNATURE_CELLS] {
    let plane = &frame.y_plane;
    let width = plane.width().get();
    let height = plane.height().get();
    let stride = plane.geometry().stride.get();
    let origin = plane.data_origin();
    let data = plane.data();
    let sample_max = (1u64 << bit_depth).saturating_sub(1).max(1);
    let mut signature = [0u8; FRAME_LUMA_SIGNATURE_CELLS];
    let mut cell_idx = 0;

    for cell_y in 0..FRAME_LUMA_SIGNATURE_ROWS {
        let y0 = cell_y * height / FRAME_LUMA_SIGNATURE_ROWS;
        let y1 = (cell_y + 1) * height / FRAME_LUMA_SIGNATURE_ROWS;
        for cell_x in 0..FRAME_LUMA_SIGNATURE_COLS {
            let x0 = cell_x * width / FRAME_LUMA_SIGNATURE_COLS;
            let x1 = (cell_x + 1) * width / FRAME_LUMA_SIGNATURE_COLS;
            let mut sum = 0u64;

            for sample_y in 0..FRAME_LUMA_SIGNATURE_SAMPLES_PER_CELL {
                let y = sampled_cell_position(
                    y0,
                    y1,
                    height,
                    sample_y,
                    FRAME_LUMA_SIGNATURE_SAMPLES_PER_CELL,
                );
                for sample_x in 0..FRAME_LUMA_SIGNATURE_SAMPLES_PER_CELL {
                    let x = sampled_cell_position(
                        x0,
                        x1,
                        width,
                        sample_x,
                        FRAME_LUMA_SIGNATURE_SAMPLES_PER_CELL,
                    );
                    let pixel = data[origin + y * stride + x]
                        .to_u32()
                        .expect("pixel value should fit in u32");
                    sum += u64::from(pixel);
                }
            }

            let samples = (FRAME_LUMA_SIGNATURE_SAMPLES_PER_CELL
                * FRAME_LUMA_SIGNATURE_SAMPLES_PER_CELL) as u64;
            let scaled = (sum * 255 + (sample_max * samples) / 2) / (sample_max * samples);
            signature[cell_idx] = scaled.min(255) as u8;
            cell_idx += 1;
        }
    }

    signature
}

fn sampled_cell_position(
    start: usize,
    end: usize,
    limit: usize,
    sample: usize,
    samples: usize,
) -> usize {
    let len = end.saturating_sub(start).max(1);
    (start + ((2 * sample + 1) * len) / (2 * samples)).min(limit.saturating_sub(1))
}

fn adjusted_peak_cost(current_cost: f64, previous_costs: impl Iterator<Item = f64>) -> f64 {
    let mut adjusted_cost: Option<f64> = None;
    for previous_cost in previous_costs {
        let cost = current_cost - previous_cost;
        adjusted_cost = Some(adjusted_cost.map_or(cost, |adjusted_cost| adjusted_cost.min(cost)));
        if cost < 0.0 {
            return 0.0;
        }
    }
    adjusted_cost.unwrap_or(0.0)
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

    fn forward_test_detector(width: usize, height: usize) -> SceneChangeDetector<u8> {
        SceneChangeDetector::new(
            (width, height),
            8,
            Rational32::new(30, 1),
            ChromaSubsampling::Yuv420,
            5,
            SceneDetectionSpeed::High,
            DetectionTuning::default(),
            0,
            0,
        )
    }

    fn fill_plane_with(plane: &mut Plane<u8>, f: impl Fn(usize, usize) -> u8) {
        let stride = plane.geometry().stride.get();
        let width = plane.width().get();
        let height = plane.height().get();
        let origin = plane.data_origin();
        let data = plane.data_mut();
        for y in 0..height {
            for x in 0..width {
                data[origin + y * stride + x] = f(y, x);
            }
        }
    }

    fn yuv420_frame(
        width: usize,
        height: usize,
        luma: impl Fn(usize, usize) -> u8,
        chroma: impl Fn(usize, usize) -> u8,
    ) -> Arc<Frame<u8>> {
        let mut frame = v_frame::frame::FrameBuilder::new(
            NonZeroUsize::new(width).unwrap(),
            NonZeroUsize::new(height).unwrap(),
            ChromaSubsampling::Yuv420,
            NonZeroU8::new(8).unwrap(),
        )
        .build::<u8>()
        .unwrap();
        fill_plane_with(&mut frame.y_plane, luma);
        if let Some(plane) = frame.u_plane.as_mut() {
            fill_plane_with(plane, &chroma);
        }
        if let Some(plane) = frame.v_plane.as_mut() {
            fill_plane_with(plane, &chroma);
        }
        Arc::new(frame)
    }

    fn forward_test_options() -> ForwardSimilarityOptions {
        ForwardSimilarityOptions {
            enabled: true,
            frames: 80,
            window_frames: 3,
            min_offset: 4,
            threshold_8bit: 6.0,
            relaxed_threshold_8bit: 8.5,
            mask_percent: 0.20,
            mask_region_cols: 8,
            mask_region_rows: 4,
            chroma_weight: 0.25,
            ..ForwardSimilarityOptions::default()
        }
    }

    // P5 parity: with the global `(1,1,false)` masks cached (Part A) and the
    // per-entry neighbour memo (Part B), every deque entry's `imp_block_cost` and
    // `global_imp_block_cost` must stay bit-identical to a brute-force recompute
    // that consults neither the mask cache nor the memo. `parallel_high_matches_
    // sequential` runs the SAME detector code on both sides, so it cannot catch a
    // logic bug here; this pins the optimized path against an independent reference.
    #[test]
    fn high_importance_refresh_matches_bruteforce() {
        // Recompute one entry's temporal top-block score from scratch — always
        // sorting (never consulting the cache) and ignoring the memo — mirroring
        // the production scorers exactly. `capped` selects the spatial path
        // (imp_block_cost) vs the global path (global_imp_block_cost).
        fn top_score(
            deque: &[ScenecutAnalysis],
            index: usize,
            capped: bool,
            previous_percent: f64,
            current_percent: f64,
            next_percent: f64,
            region_cols: usize,
            region_rows: usize,
        ) -> Option<f64> {
            let current = deque.get(index)?;
            let block_count = current.importance_blocks.len();
            if block_count == 0 {
                return None;
            }
            if capped && (current.importance_cols == 0 || current.importance_rows == 0) {
                return None;
            }
            let mut selected = vec![false; block_count];
            let mark = |src: &[f64], percent: f64, sel: &mut [bool]| {
                if capped {
                    mark_top_blocks_spatially_capped(
                        src,
                        percent,
                        current.importance_cols,
                        current.importance_rows,
                        region_cols,
                        region_rows,
                        sel,
                    );
                } else {
                    mark_top_blocks(src, percent, sel);
                }
            };
            if let Some(previous) = deque.get(index + 1) {
                mark(&previous.importance_blocks, previous_percent, &mut selected);
            }
            mark(&current.importance_blocks, current_percent, &mut selected);
            if index > 0
                && let Some(next) = deque.get(index - 1)
            {
                mark(&next.importance_blocks, next_percent, &mut selected);
            }
            let mut total = 0.0;
            let mut count = 0usize;
            for (i, &is_selected) in selected.iter().enumerate() {
                if is_selected {
                    total += current.importance_blocks[i];
                    count += 1;
                }
            }
            (count > 0).then_some(total / count as f64)
        }

        let (previous_percent, current_percent, next_percent) = (0.10_f64, 0.15_f64, 0.10_f64);
        let (region_cols, region_rows) = (8usize, 4usize);
        let (w, h) = (256, 128);

        let mut tuning = DetectionTuning::high_quality();
        tuning.importance_aggregation = ImportanceAggregation::SpatialTemporalTopBlocks {
            previous_percent,
            current_percent,
            next_percent,
            region_cols,
            region_rows,
        };
        // Drive the deque purely by insert/pop; no forward-similarity frame reads.
        tuning.forward_similarity.enabled = false;

        let mut det = SceneChangeDetector::<u8>::new(
            (w, h),
            8,
            Rational32::new(30, 1),
            ChromaSubsampling::Yuv420,
            5,
            SceneDetectionSpeed::High,
            tuning,
            0,
            1000,
        );

        // Varied content so different blocks win the selection on different frames
        // (otherwise the masks are trivially equal and the test proves nothing).
        let frames: Vec<Arc<Frame<u8>>> = (0..24)
            .map(|t| {
                yuv420_frame(
                    w,
                    h,
                    move |y, x| {
                        let stripe = (x + y * 2 + t * 7) as u32 % 64;
                        let patch = if x / 16 == (t * 3) % (w / 16) && y / 16 == t % (h / 16) {
                            210
                        } else {
                            0
                        };
                        ((stripe + patch) % 256) as u8
                    },
                    |_, _| 128,
                )
            })
            .collect();

        // Mimic the serial driver: window starts at frameno-1 and spans the
        // lookahead; input_frameno == frameno. Exercises init, warmup, the
        // steady-state back-pop, and the end-of-video tail (window shrinks).
        let lookahead = 5usize;
        let mut saw_divergence = false;
        for frameno in 1..frames.len() {
            let start = frameno - 1;
            let end = (start + lookahead + 2).min(frames.len());
            let window: Vec<&Arc<Frame<u8>>> = frames[start..end].iter().collect();
            if window.len() < 2 {
                break;
            }
            det.analyze_next_frame(&window, frameno, 0);

            // The end-of-frame `pop()` runs after `adaptive_scenecut`, so it can
            // leave the new oldest entry's stored score reflecting the neighbour it
            // just lost. The detector never consumes that entry until the next
            // frame's `adaptive_scenecut`, whose top-of-function refresh runs first
            // (the entry's `(prev_present, ..)` key flipped, forcing a recompute).
            // Mirror that refresh here so we compare against the values the
            // algorithm actually reads — and so a wrongly-skipped refresh (which
            // this pass would also skip, by the same memo key) still surfaces as a
            // mismatch against the brute-force reference below.
            for idx in 0..det.score_deque.len() {
                det.refresh_importance_metrics(idx);
            }

            for idx in 0..det.score_deque.len() {
                let raw = det.score_deque[idx].result.imp_block_cost_raw;
                let want_imp = top_score(
                    &det.score_deque,
                    idx,
                    true,
                    previous_percent,
                    current_percent,
                    next_percent,
                    region_cols,
                    region_rows,
                )
                .unwrap_or(raw);
                let want_global = top_score(
                    &det.score_deque,
                    idx,
                    false,
                    previous_percent,
                    current_percent,
                    next_percent,
                    region_cols,
                    region_rows,
                )
                .unwrap_or(raw);

                let got = det.score_deque[idx].result;
                assert_eq!(
                    got.imp_block_cost.to_bits(),
                    want_imp.to_bits(),
                    "imp_block_cost mismatch: frameno={frameno} idx={idx}"
                );
                assert_eq!(
                    got.global_imp_block_cost.to_bits(),
                    want_global.to_bits(),
                    "global_imp_block_cost mismatch: frameno={frameno} idx={idx}"
                );
                if (want_imp - want_global).abs() > 1e-9 {
                    saw_divergence = true;
                }
            }
        }
        assert!(
            saw_divergence,
            "content too uniform: capped and global scores never diverged"
        );
    }

    // The exact masked-luma metric (`masked_luma_delta_8bit`) only samples a 1/16
    // lattice inside full 32x32 blocks, so a pair can be identical on the lattice
    // (masked delta 0) while the full-resolution SAD is huge. The removed
    // `masked_delta_lower_bound_8bit` early-out derived its "lower bound" from that
    // full SAD and would reject such a pair; Variant C uses the exact masked metric
    // and must accept it. This also exercises gpt5.5 counterexample #1.
    #[test]
    fn forward_segment_uses_exact_masked_metric_not_full_sad() {
        let (w, h) = (96, 64);
        let options = forward_test_options();
        let f1 = yuv420_frame(w, h, |_, _| 0, |_, _| 128);
        // Lattice points (y%4==0 && x%4==0) stay 0; everything else jumps to 255.
        let f2 = yuv420_frame(
            w,
            h,
            |y, x| if y % 4 == 0 && x % 4 == 0 { 0 } else { 255 },
            |_, _| 128,
        );
        let mut det = forward_test_detector(w, h);
        let masked = det.masked_luma_delta_8bit(&f1, &f2, options.mask_percent, 8, 4);
        let full = det.luma_delta_8bit(&f1, &f2);
        assert!(masked < 1.0, "masked metric should see a near-identical pair");
        assert!(full > 200.0, "full SAD should see a very different pair");
        // The segment delta must follow the masked metric (accept), not the SAD.
        let got = det.forward_segment_similarity_delta_8bit(&[&f1], &[&f2], 0, 0, options, 8.5);
        assert!(
            got < 1.0,
            "segment delta must track the exact masked metric, got {got}"
        );
    }

    // For every offset the reported delta must equal an always-exact reference
    // (`masked_luma + weighted_chroma`) whenever it is at/below the confirmation
    // threshold (so every `candidate.delta <= K` consumer sees the exact value),
    // and must imply the exact value is also above when it is reported above.
    #[test]
    fn forward_segment_matches_always_exact_reference() {
        let (w, h) = (96, 64);
        let options = forward_test_options();
        let pairs = [
            // identical luma + chroma
            (
                yuv420_frame(w, h, |_, _| 100, |_, _| 128),
                yuv420_frame(w, h, |_, _| 100, |_, _| 128),
            ),
            // shifted gradient (small-ish luma + chroma change)
            (
                yuv420_frame(w, h, |y, x| ((y + x) % 256) as u8, |_, _| 128),
                yuv420_frame(w, h, |y, x| ((y + x + 30) % 256) as u8, |_, _| 150),
            ),
            // lattice-identical luma, huge off-lattice change (masked ~0)
            (
                yuv420_frame(w, h, |_, _| 0, |_, _| 128),
                yuv420_frame(
                    w,
                    h,
                    |y, x| if y % 4 == 0 && x % 4 == 0 { 0 } else { 255 },
                    |_, _| 128,
                ),
            ),
            // chroma-only difference (identical luma -> luma prefilter must not skip)
            (
                yuv420_frame(w, h, |_, _| 80, |_, _| 100),
                yuv420_frame(w, h, |_, _| 80, |_, _| 200),
            ),
            // large luma difference (rejected high above threshold)
            (
                yuv420_frame(w, h, |_, _| 10, |_, _| 128),
                yuv420_frame(w, h, |_, _| 240, |_, _| 128),
            ),
        ];
        for &confirmation in &[6.0_f64, 8.5, 50.0] {
            for (f1, f2) in &pairs {
                let mut det = forward_test_detector(w, h);
                let got = det
                    .forward_segment_similarity_delta_8bit(&[f1], &[f2], 0, 0, options, confirmation);
                let exact_luma =
                    det.masked_luma_delta_8bit(f1, f2, options.mask_percent, 8, 4);
                let exact_chroma = det.weighted_chroma_delta_8bit(f1, f2, options.chroma_weight);
                let exact = exact_luma + exact_chroma;
                if got <= confirmation {
                    assert_eq!(
                        got.to_bits(),
                        exact.to_bits(),
                        "consumable delta must be bit-exact (conf {confirmation})"
                    );
                } else {
                    assert!(
                        exact > confirmation,
                        "a delta reported above {confirmation} must be exactly above it (exact {exact})"
                    );
                }
            }
        }
    }

    fn fill_plane_generic<T: Pixel>(plane: &mut Plane<T>, f: &impl Fn(usize, usize) -> i32) {
        let stride = plane.geometry().stride.get();
        let width = plane.width().get();
        let height = plane.height().get();
        let origin = plane.data_origin();
        let data = plane.data_mut();
        for y in 0..height {
            for x in 0..width {
                data[origin + y * stride + x] = T::from(f(y, x)).unwrap();
            }
        }
    }

    fn yuv420_frame_generic<T: Pixel>(
        width: usize,
        height: usize,
        bit_depth: u8,
        luma: impl Fn(usize, usize) -> i32,
        chroma: impl Fn(usize, usize) -> i32,
    ) -> Arc<Frame<T>> {
        let mut frame = v_frame::frame::FrameBuilder::new(
            NonZeroUsize::new(width).unwrap(),
            NonZeroUsize::new(height).unwrap(),
            ChromaSubsampling::Yuv420,
            NonZeroU8::new(bit_depth).unwrap(),
        )
        .build::<T>()
        .unwrap();
        fill_plane_generic(&mut frame.y_plane, &luma);
        if let Some(plane) = frame.u_plane.as_mut() {
            fill_plane_generic(plane, &chroma);
        }
        if let Some(plane) = frame.v_plane.as_mut() {
            fill_plane_generic(plane, &chroma);
        }
        Arc::new(frame)
    }

    fn forward_test_detector_bits<T: Pixel>(
        width: usize,
        height: usize,
        bit_depth: usize,
    ) -> SceneChangeDetector<T> {
        SceneChangeDetector::new(
            (width, height),
            bit_depth,
            Rational32::new(30, 1),
            ChromaSubsampling::Yuv420,
            5,
            SceneDetectionSpeed::High,
            DetectionTuning::default(),
            0,
            0,
        )
    }

    // Core Variant D invariant: the cached block-sum bound is never above the exact
    // masked-luma metric, across bit depths, dimensions, masks, region modes, and
    // pixel patterns (including sign-cancelling and lattice-aligned ones).
    fn check_prefilter_lower_bound<T: Pixel>(bit_depth: usize) {
        let max = (1i32 << bit_depth) - 1;
        let half = max / 2;
        let quarter = max / 4;
        type Pat = Box<dyn Fn(usize, usize) -> i32>;
        let dims = [(64usize, 64usize), (96, 64), (100, 70)];
        let masks = [0.01f64, 0.20, 0.95];
        let regions = [(1usize, 1usize), (8usize, 4usize)];
        for &(w, h) in &dims {
            let cases: Vec<(&str, Pat, Pat)> = vec![
                ("identical", Box::new(move |_, _| half), Box::new(move |_, _| half)),
                ("uniform_max", Box::new(|_, _| 0), Box::new(move |_, _| max)),
                (
                    "gradient",
                    Box::new(|_, _| 0),
                    Box::new(move |y, x| (((y + x) as i32) * 7) % (max + 1)),
                ),
                (
                    "checker",
                    Box::new(|_, _| 0),
                    Box::new(move |y, x| if (x / 4 + y / 4) % 2 == 0 { 0 } else { max }),
                ),
                (
                    "lattice_identical",
                    Box::new(|_, _| 0),
                    Box::new(move |y, x| if y % 4 == 0 && x % 4 == 0 { 0 } else { max }),
                ),
                (
                    "sign_cancel",
                    Box::new(move |_, _| half),
                    Box::new(move |y, x| {
                        if (x / 4 + y / 4) % 2 == 0 {
                            (half + quarter).min(max)
                        } else {
                            (half - quarter).max(0)
                        }
                    }),
                ),
            ];
            for (name, f1_fn, f2_fn) in &cases {
                let f1: Arc<Frame<T>> =
                    yuv420_frame_generic(w, h, bit_depth as u8, |y, x| f1_fn(y, x), |_, _| half);
                let f2: Arc<Frame<T>> =
                    yuv420_frame_generic(w, h, bit_depth as u8, |y, x| f2_fn(y, x), |_, _| half);
                for &mask in &masks {
                    for &(rc, rr) in &regions {
                        let mut det = forward_test_detector_bits::<T>(w, h, bit_depth);
                        det.forward_ref_signatures
                            .push(build_forward_prefilter_signature(&f1));
                        det.forward_post_signatures
                            .push(Some(build_forward_prefilter_signature(&f2)));
                        let bound = det
                            .forward_segment_prefilter_bound(&[&f2], 0, 0, 1, mask)
                            .expect("full block grid present");
                        let exact = det.masked_luma_delta_8bit(&f1, &f2, mask, rc, rr);
                        assert!(
                            bound <= exact + 1e-9,
                            "{name} {w}x{h} bd{bit_depth} mask {mask} region {rc}x{rr}: \
                             bound {bound} exceeded exact {exact}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn forward_prefilter_bound_is_below_exact_masked_luma() {
        check_prefilter_lower_bound::<u8>(8);
        check_prefilter_lower_bound::<u16>(10);
    }

    // The prefilter must never alter a consumable (<= confirmation) delta: those
    // are returned exactly by the exact path whether or not signatures are present.
    #[test]
    fn forward_prefilter_preserves_consumable_deltas() {
        let (w, h) = (96, 64);
        let options = forward_test_options();
        let pairs = [
            (
                yuv420_frame(w, h, |_, _| 100, |_, _| 128),
                yuv420_frame(w, h, |_, _| 100, |_, _| 128),
            ),
            (
                yuv420_frame(w, h, |_, _| 80, |_, _| 100),
                yuv420_frame(w, h, |_, _| 80, |_, _| 200),
            ),
            (
                yuv420_frame(w, h, |_, _| 10, |_, _| 128),
                yuv420_frame(w, h, |_, _| 240, |_, _| 128),
            ),
            (
                yuv420_frame(w, h, |y, x| ((y + x) % 256) as u8, |_, _| 128),
                yuv420_frame(w, h, |y, x| ((y + x + 14) % 256) as u8, |_, _| 150),
            ),
        ];
        for &confirmation in &[6.0_f64, 8.5, 50.0] {
            for (f1, f2) in &pairs {
                let mut without = forward_test_detector(w, h);
                let got_without = without
                    .forward_segment_similarity_delta_8bit(&[f1], &[f2], 0, 0, options, confirmation);

                let mut with = forward_test_detector(w, h);
                with.forward_ref_signatures
                    .push(build_forward_prefilter_signature(f1));
                with.forward_post_signatures
                    .push(Some(build_forward_prefilter_signature(f2)));
                let got_with = with
                    .forward_segment_similarity_delta_8bit(&[f1], &[f2], 0, 0, options, confirmation);

                if got_without <= confirmation {
                    assert_eq!(
                        got_without.to_bits(),
                        got_with.to_bits(),
                        "prefilter changed a consumable delta (conf {confirmation})"
                    );
                } else {
                    assert!(
                        got_with > confirmation,
                        "prefilter reported a consumable value for a rejected pair"
                    );
                }
            }
        }
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
            relaxed_max_cost_ratio: 3.0,
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
        assert_eq!(
            forward_similarity_threshold_8bit(options, score_with_ratios(3.1, 5.0)),
            6.0
        );
    }

    #[test]
    fn forward_similarity_rejects_dominant_return_candidate_for_relaxed_match() {
        let options = ForwardSimilarityOptions {
            threshold_8bit: 6.0,
            relaxed_threshold_8bit: 7.5,
            relaxed_min_cost_ratio: 1.0,
            relaxed_max_cost_ratio: 3.0,
            relaxed_importance_min_cost_ratio: 0.45,
            relaxed_min_imp_block_ratio: 3.6,
            relaxed_max_return_cost_ratio: 2.5,
            relaxed_max_return_cost_ratio_multiplier: 4.0,
            ..ForwardSimilarityOptions::default()
        };
        assert!(forward_similarity_return_candidate_allowed(
            options,
            score_with_ratios(1.1, 3.0),
            score_with_ratios(2.4, 3.0),
            7.1
        ));
        assert!(!forward_similarity_return_candidate_allowed(
            options,
            score_with_ratios(0.9, 4.0),
            score_with_ratios(9.0, 3.0),
            7.1
        ));
        assert!(forward_similarity_return_candidate_allowed(
            options,
            score_with_ratios(0.9, 4.0),
            score_with_ratios(9.0, 3.0),
            5.9
        ));
    }

    #[test]
    fn forward_similarity_allows_strict_hard_flash_without_return_candidate() {
        let options = ForwardSimilarityOptions {
            flash_return_without_candidate: true,
            flash_return_frames: 40,
            flash_return_min_cost_ratio: 1.0,
            threshold_8bit: 6.0,
            ..ForwardSimilarityOptions::default()
        };
        assert!(forward_similarity_flash_return_without_candidate_allowed(
            options,
            score_with_ratios(1.2, 3.0),
            30,
            5.9
        ));
        assert!(!forward_similarity_flash_return_without_candidate_allowed(
            options,
            score_with_ratios(0.9, 3.0),
            30,
            5.9
        ));
        assert!(!forward_similarity_flash_return_without_candidate_allowed(
            options,
            score_with_ratios(1.2, 3.0),
            41,
            5.9
        ));
        assert!(!forward_similarity_flash_return_without_candidate_allowed(
            options,
            score_with_ratios(1.2, 3.0),
            30,
            6.1
        ));
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

fn mark_cached_top_blocks(mask: &[bool], selected: &mut [bool]) {
    for (idx, &is_selected) in mask.iter().enumerate() {
        if is_selected && let Some(slot) = selected.get_mut(idx) {
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
    SuppressedTextCardCluster,
    SuppressedFastMotion,
    SuppressedStaticCredits,
    SuppressedDarkOcclusion,
    CutDarkScenePeak,
    CutSparseScenePeak,
    CutRefinedSparsePeak,
    SuppressedRefinedSparsePeak,
    CutShiftedBoundary,
    SuppressedShiftedBoundary,
    CutAbaReturn,
    SuppressedAbaChain,
    CutForwardSimilarityRecovery,
}

/// Contains the scores for scenecut analysis on a single frame
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct ScenecutResult {
    pub inter_cost: f64,
    pub motion_inter_cost: f64,
    pub motion_cost_computed: bool,
    pub imp_block_cost_raw: f64,
    pub imp_block_cost: f64,
    pub global_imp_block_cost: f64,
    pub imp_block_threshold: f64,
    pub imp_block_ratio: f64,
    pub global_imp_block_ratio: f64,
    pub backward_adjusted_cost: f64,
    pub forward_adjusted_cost: f64,
    pub motion_backward_adjusted_cost: f64,
    pub motion_forward_adjusted_cost: f64,
    pub threshold: f64,
    pub cost_ratio: f64,
    pub motion_cost_ratio: f64,
    pub avg_luma_8bit: f64,
    pub static_bad_block_ratio: f64,
    pub static_good_block_ratio: f64,
    pub me_bad_block_ratio: f64,
    pub me_good_block_ratio: f64,
    #[cfg_attr(
        feature = "serialize",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub frame_luma_signature: Option<[u8; FRAME_LUMA_SIGNATURE_CELLS]>,
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
            motion_inter_cost: inter_cost,
            motion_cost_computed: false,
            imp_block_cost_raw: imp_block_cost,
            imp_block_cost,
            global_imp_block_cost: imp_block_cost,
            imp_block_threshold,
            imp_block_ratio: 0.0,
            global_imp_block_ratio: 0.0,
            backward_adjusted_cost: 0.0,
            forward_adjusted_cost: 0.0,
            motion_backward_adjusted_cost: 0.0,
            motion_forward_adjusted_cost: 0.0,
            threshold,
            cost_ratio: 0.0,
            motion_cost_ratio: 0.0,
            avg_luma_8bit,
            static_bad_block_ratio: 0.0,
            static_good_block_ratio: 0.0,
            me_bad_block_ratio: 0.0,
            me_good_block_ratio: 0.0,
            frame_luma_signature: None,
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
        self.motion_cost_ratio = if self.threshold > 0.0 {
            self.motion_forward_adjusted_cost / self.threshold
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
