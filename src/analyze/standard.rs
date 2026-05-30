use std::sync::Arc;

use v_frame::{frame::Frame, pixel::Pixel};

use super::{SceneChangeDetector, ScenecutAnalysis, ScenecutResult};
use crate::{
    analyze::{
        importance::estimate_importance_block_difference_detailed,
        inter::estimate_inter_costs_detailed,
        intra::estimate_intra_costs,
    },
    data::motion::FrameMEStats,
    math::Fixed,
};

impl<T: Pixel> SceneChangeDetector<T> {
    /// Run a comparison between two frames to determine if they qualify for a
    /// scenecut.
    ///
    /// We gather both intra and inter costs for the frames,
    /// as well as an importance-block-based difference,
    /// and use all three metrics.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(input_frameno))
    )]
    pub(super) fn cost_scenecut(
        &mut self,
        frame1: &Arc<Frame<T>>,
        frame2: &Arc<Frame<T>>,
        input_frameno: usize,
    ) -> ScenecutAnalysis {
        let mut intra_cost = 0.0;
        let mut mv_inter_cost = None;
        let mut imp_block_diff = None;

        let cols = 2 * self.resolution.0.align_power_of_two_and_shift(3);
        let rows = 2 * self.resolution.1.align_power_of_two_and_shift(3);

        let buffer = if let Some(buffer) = &self.frame_me_stats_buffer {
            Arc::clone(buffer)
        } else {
            let frame_me_stats = FrameMEStats::new_arc_array(cols, rows);
            let clone = Arc::clone(&frame_me_stats);
            self.frame_me_stats_buffer = Some(frame_me_stats);
            clone
        };

        rayon::scope(|s| {
            s.spawn(|_| {
                let temp_plane = self
                    .temp_plane
                    .get_or_insert_with(|| frame2.y_plane.clone());

                let intra_costs = estimate_intra_costs(temp_plane, frame2, self.bit_depth);
                if let Some(ref mut intra_cache) = self.intra_costs {
                    intra_cache.insert(input_frameno, intra_costs.clone());
                }

                intra_cost = intra_costs.iter().map(|&cost| cost as u64).sum::<u64>() as f64
                    / intra_costs.len() as f64;
            });
            s.spawn(|_| {
                mv_inter_cost = Some(estimate_inter_costs_detailed(
                    frame2,
                    frame1,
                    self.bit_depth,
                    self.frame_rate,
                    self.chroma_sampling,
                    buffer,
                ));
            });
            s.spawn(|_| {
                imp_block_diff = Some(estimate_importance_block_difference_detailed(
                    frame2,
                    frame1,
                    self.bit_depth,
                ));
            });
        });

        // `BIAS` determines how likely we are
        // to choose a keyframe, between 0.0-1.0.
        // Higher values mean we are more likely to choose a keyframe.
        // This value was chosen based on trials using the new
        // adaptive scenecut code.
        const BIAS: f64 = 0.7;
        let threshold = intra_cost * (1.0 - BIAS);
        let imp_block_diff = imp_block_diff.expect("importance block diff should be set");
        let mv_inter_cost = mv_inter_cost.expect("inter cost should be set");

        let mut result = ScenecutResult::new(
            mv_inter_cost.mean,
            imp_block_diff.mean,
            self.importance_threshold(imp_block_diff.avg_luma_8bit),
            threshold,
            imp_block_diff.avg_luma_8bit,
        );
        result.me_bad_block_ratio = mv_inter_cost.me_bad_block_ratio;
        result.me_good_block_ratio = mv_inter_cost.me_good_block_ratio;

        ScenecutAnalysis {
            result,
            importance_blocks: imp_block_diff.blocks,
            importance_cols: imp_block_diff.cols,
            importance_rows: imp_block_diff.rows,
        }
    }

    #[cfg(test)]
    fn cost_scenecut_sequential(
        &mut self,
        frame1: &Arc<Frame<T>>,
        frame2: &Arc<Frame<T>>,
        input_frameno: usize,
    ) -> ScenecutAnalysis {
        let cols = 2 * self.resolution.0.align_power_of_two_and_shift(3);
        let rows = 2 * self.resolution.1.align_power_of_two_and_shift(3);

        let buffer = if let Some(buffer) = &self.frame_me_stats_buffer {
            Arc::clone(buffer)
        } else {
            let frame_me_stats = FrameMEStats::new_arc_array(cols, rows);
            let clone = Arc::clone(&frame_me_stats);
            self.frame_me_stats_buffer = Some(frame_me_stats);
            clone
        };

        let temp_plane = self
            .temp_plane
            .get_or_insert_with(|| frame2.y_plane.clone());
        let intra_costs = estimate_intra_costs(temp_plane, frame2, self.bit_depth);
        if let Some(ref mut intra_cache) = self.intra_costs {
            intra_cache.insert(input_frameno, intra_costs.clone());
        }
        let intra_cost = intra_costs.iter().map(|&cost| cost as u64).sum::<u64>() as f64
            / intra_costs.len() as f64;

        let mv_inter_cost = estimate_inter_costs_detailed(
            frame2,
            frame1,
            self.bit_depth,
            self.frame_rate,
            self.chroma_sampling,
            buffer,
        );
        let imp_block_diff =
            estimate_importance_block_difference_detailed(frame2, frame1, self.bit_depth);

        const BIAS: f64 = 0.7;
        let threshold = intra_cost * (1.0 - BIAS);

        let mut result = ScenecutResult::new(
            mv_inter_cost.mean,
            imp_block_diff.mean,
            self.importance_threshold(imp_block_diff.avg_luma_8bit),
            threshold,
            imp_block_diff.avg_luma_8bit,
        );
        result.me_bad_block_ratio = mv_inter_cost.me_bad_block_ratio;
        result.me_good_block_ratio = mv_inter_cost.me_good_block_ratio;

        ScenecutAnalysis {
            result,
            importance_blocks: imp_block_diff.blocks,
            importance_cols: imp_block_diff.cols,
            importance_rows: imp_block_diff.rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        num::{NonZeroU8, NonZeroUsize},
        sync::Arc,
    };

    use num_rational::Rational32;
    use v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder, plane::Plane};

    use super::*;
    use crate::{DetectionTuning, SceneDetectionSpeed};

    fn fill_plane<T: Pixel>(plane: &mut Plane<T>, seed: usize) {
        let stride = plane.geometry().stride.get();
        let width = plane.width().get();
        let height = plane.height().get();
        let origin = plane.data_origin();
        let data = plane.data_mut();
        for row in 0..height {
            for col in 0..width {
                let value = ((row * 17 + col * 13 + seed) % 256) as i32;
                data[origin + row * stride + col] = T::from(value).expect("valid pixel value");
            }
        }
    }

    fn test_frame(seed: usize) -> Arc<Frame<u8>> {
        let mut frame = FrameBuilder::new(
            NonZeroUsize::new(128).expect("nonzero width"),
            NonZeroUsize::new(128).expect("nonzero height"),
            ChromaSubsampling::Yuv420,
            NonZeroU8::new(8).expect("nonzero bit depth"),
        )
        .build::<u8>()
        .expect("valid test frame");

        fill_plane(&mut frame.y_plane, seed);
        if let Some(plane) = frame.u_plane.as_mut() {
            fill_plane(plane, seed + 31);
        }
        if let Some(plane) = frame.v_plane.as_mut() {
            fill_plane(plane, seed + 62);
        }

        Arc::new(frame)
    }

    fn test_detector() -> SceneChangeDetector<u8> {
        SceneChangeDetector::new(
            (128, 128),
            8,
            Rational32::new(1, 24),
            ChromaSubsampling::Yuv420,
            5,
            SceneDetectionSpeed::Standard,
            DetectionTuning::default(),
            0,
            u32::MAX as usize,
        )
    }

    fn assert_results_equal(left: ScenecutResult, right: ScenecutResult) {
        assert_eq!(left.inter_cost, right.inter_cost);
        assert_eq!(left.imp_block_cost_raw, right.imp_block_cost_raw);
        assert_eq!(left.imp_block_cost, right.imp_block_cost);
        assert_eq!(left.global_imp_block_cost, right.global_imp_block_cost);
        assert_eq!(left.imp_block_threshold, right.imp_block_threshold);
        assert_eq!(left.imp_block_ratio, right.imp_block_ratio);
        assert_eq!(left.global_imp_block_ratio, right.global_imp_block_ratio);
        assert_eq!(left.backward_adjusted_cost, right.backward_adjusted_cost);
        assert_eq!(left.forward_adjusted_cost, right.forward_adjusted_cost);
        assert_eq!(left.threshold, right.threshold);
        assert_eq!(left.cost_ratio, right.cost_ratio);
        assert_eq!(left.avg_luma_8bit, right.avg_luma_8bit);
        assert_eq!(left.me_bad_block_ratio, right.me_bad_block_ratio);
        assert_eq!(left.me_good_block_ratio, right.me_good_block_ratio);
        assert_eq!(
            left.transient_similarity_score,
            right.transient_similarity_score
        );
        assert_eq!(
            left.forward_similarity_score,
            right.forward_similarity_score
        );
        assert_eq!(
            format!("{:?}", left.forward_similarity_candidates),
            format!("{:?}", right.forward_similarity_candidates)
        );
        assert_eq!(
            left.forward_similarity_rejected_candidates,
            right.forward_similarity_rejected_candidates
        );
        assert_eq!(left.decision, right.decision);
        assert_eq!(left.forward_return_frame, right.forward_return_frame);
    }

    #[test]
    fn parallel_cost_scenecut_matches_sequential_reference() {
        let frame1 = test_frame(11);
        let frame2 = test_frame(67);
        let mut parallel = test_detector();
        let mut sequential = test_detector();

        let parallel_analysis = parallel.cost_scenecut(&frame1, &frame2, 9);
        let sequential_analysis = sequential.cost_scenecut_sequential(&frame1, &frame2, 9);

        assert_results_equal(parallel_analysis.result, sequential_analysis.result);
        assert_eq!(
            parallel_analysis.importance_blocks,
            sequential_analysis.importance_blocks
        );
        assert_eq!(
            parallel_analysis.importance_cols,
            sequential_analysis.importance_cols
        );
        assert_eq!(
            parallel_analysis.importance_rows,
            sequential_analysis.importance_rows
        );
    }
}
