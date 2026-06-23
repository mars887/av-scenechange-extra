use std::sync::Arc;

use v_frame::{frame::Frame, pixel::Pixel};

use super::{SceneChangeDetector, ScenecutAnalysis, ScenecutResult};
use crate::{
    SceneDetectionSpeed,
    analyze::{
        frame_luma_signature_8bit,
        importance::estimate_importance_block_difference_detailed,
        inter::{estimate_inter_costs_detailed, estimate_static_inter_and_importance_detailed},
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

        let compute_motion_cost = self.tuning.motion_cost_diagnostics;
        let mut motion_buffer = None;
        if compute_motion_cost {
            let cols = 2 * self.resolution.0.align_power_of_two_and_shift(3);
            let rows = 2 * self.resolution.1.align_power_of_two_and_shift(3);
            motion_buffer = Some(if let Some(buffer) = &self.frame_me_stats_buffer {
                Arc::clone(buffer)
            } else {
                let frame_me_stats = FrameMEStats::new_arc_array(cols, rows);
                let clone = Arc::clone(&frame_me_stats);
                self.frame_me_stats_buffer = Some(frame_me_stats);
                clone
            });
        }

        if self.use_cost_parallelism {
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
                if let Some(buffer) = motion_buffer {
                    // Diagnostics path: full motion estimation cannot be fused
                    // with the importance pass, so keep them as separate tasks.
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
                } else {
                    // Default path: static-SATD and importance traverse the same
                    // 8x8 grid, so fuse them into one task to halve plane reads.
                    s.spawn(|_| {
                        let (inter, imp) = estimate_static_inter_and_importance_detailed(
                            frame2,
                            frame1,
                            self.bit_depth,
                        );
                        mv_inter_cost = Some(inter);
                        imp_block_diff = Some(imp);
                    });
                }
            });
        } else {
            {
                let temp_plane = self
                    .temp_plane
                    .get_or_insert_with(|| frame2.y_plane.clone());

                let intra_costs = estimate_intra_costs(temp_plane, frame2, self.bit_depth);
                if let Some(ref mut intra_cache) = self.intra_costs {
                    intra_cache.insert(input_frameno, intra_costs.clone());
                }

                intra_cost = intra_costs.iter().map(|&cost| cost as u64).sum::<u64>() as f64
                    / intra_costs.len() as f64;
            }
            if let Some(buffer) = motion_buffer {
                // Diagnostics path: full motion estimation + separate importance.
                mv_inter_cost = Some(estimate_inter_costs_detailed(
                    frame2,
                    frame1,
                    self.bit_depth,
                    self.frame_rate,
                    self.chroma_sampling,
                    buffer,
                ));
                imp_block_diff = Some(estimate_importance_block_difference_detailed(
                    frame2,
                    frame1,
                    self.bit_depth,
                ));
            } else {
                // Default path: one fused pass over the shared 8x8 grid.
                let (inter, imp) =
                    estimate_static_inter_and_importance_detailed(frame2, frame1, self.bit_depth);
                mv_inter_cost = Some(inter);
                imp_block_diff = Some(imp);
            }
        }

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
        result.motion_inter_cost = mv_inter_cost.motion_mean;
        result.motion_cost_computed = mv_inter_cost.motion_cost_computed;
        result.static_bad_block_ratio = mv_inter_cost.static_bad_block_ratio;
        result.static_good_block_ratio = mv_inter_cost.static_good_block_ratio;
        result.me_bad_block_ratio = mv_inter_cost.me_bad_block_ratio;
        result.me_good_block_ratio = mv_inter_cost.me_good_block_ratio;
        // S3: the 32-cell luma signature is consumed only by High-mode
        // postprocess passes (apply_scenechange_postprocess, gated on
        // analysis_speed == High). Skip it otherwise — pure overhead in Standard.
        result.frame_luma_signature = if self.scene_detection_mode == SceneDetectionSpeed::High {
            Some(frame_luma_signature_8bit(frame2, self.bit_depth))
        } else {
            None
        };

        ScenecutAnalysis {
            result,
            importance_blocks: imp_block_diff.blocks,
            importance_cols: imp_block_diff.cols,
            importance_rows: imp_block_diff.rows,
            top_block_masks: Default::default(),
            importance_neighbor_state: None,
        }
    }
}
