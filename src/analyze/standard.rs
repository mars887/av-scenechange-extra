use std::sync::Arc;

use v_frame::{frame::Frame, pixel::Pixel};

use super::{SceneChangeDetector, ScenecutAnalysis, ScenecutResult};
use crate::{
    SceneDetectionSpeed,
    analyze::{
        frame_luma_signature_8bit,
        importance::{IMPORTANCE_BLOCK_SIZE, estimate_importance_block_difference_detailed},
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
        let mut intra_block_costs = None;
        let mut mv_inter_cost = None;
        let mut imp_block_diff = None;
        let mut inter_block_costs = None;

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

                    intra_block_costs = Some(intra_costs);
                });
                if let Some(buffer) = motion_buffer {
                    // Diagnostics path: full motion estimation cannot be fused
                    // with the importance pass, so keep them as separate tasks.
                    s.spawn(|_| {
                        let (inter, static_blocks) = estimate_inter_costs_detailed(
                            frame2,
                            frame1,
                            self.bit_depth,
                            self.frame_rate,
                            self.chroma_sampling,
                            buffer,
                        );
                        mv_inter_cost = Some(inter);
                        inter_block_costs = Some(static_blocks);
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
                        let (inter, imp, static_blocks) =
                            estimate_static_inter_and_importance_detailed(
                                frame2,
                                frame1,
                                self.bit_depth,
                            );
                        mv_inter_cost = Some(inter);
                        imp_block_diff = Some(imp);
                        inter_block_costs = Some(static_blocks);
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

                intra_block_costs = Some(intra_costs);
            }
            if let Some(buffer) = motion_buffer {
                // Diagnostics path: full motion estimation + separate importance.
                let (inter, static_blocks) = estimate_inter_costs_detailed(
                    frame2,
                    frame1,
                    self.bit_depth,
                    self.frame_rate,
                    self.chroma_sampling,
                    buffer,
                );
                mv_inter_cost = Some(inter);
                inter_block_costs = Some(static_blocks);
                imp_block_diff = Some(estimate_importance_block_difference_detailed(
                    frame2,
                    frame1,
                    self.bit_depth,
                ));
            } else {
                // Default path: one fused pass over the shared 8x8 grid.
                let (inter, imp, static_blocks) =
                    estimate_static_inter_and_importance_detailed(frame2, frame1, self.bit_depth);
                mv_inter_cost = Some(inter);
                imp_block_diff = Some(imp);
                inter_block_costs = Some(static_blocks);
            }
        }

        let imp_block_diff = imp_block_diff.expect("importance block diff should be set");
        let mv_inter_cost = mv_inter_cost.expect("inter cost should be set");
        let intra_block_costs = intra_block_costs.expect("intra costs should be set");

        // The intra grid is the same importance-block grid, so the threshold
        // must aggregate over the same active (non-bar) region as the inter
        // cost — otherwise bars would dilate one side of the cost ratio only.
        let active_region = mv_inter_cost.active_region;
        // Both estimators observe the same frame pair, so their independently
        // detected regions must agree (the fused path shares one detection).
        debug_assert_eq!(imp_block_diff.active_region, active_region);
        let intra_cost = if active_region.is_full() {
            intra_block_costs.iter().map(|&cost| cost as u64).sum::<u64>() as f64
                / intra_block_costs.len() as f64
        } else {
            active_region
                .indices()
                .map(|idx| intra_block_costs[idx] as u64)
                .sum::<u64>() as f64
                / active_region.count() as f64
        };

        // `BIAS` determines how likely we are
        // to choose a keyframe, between 0.0-1.0.
        // Higher values mean we are more likely to choose a keyframe.
        // This value was chosen based on trials using the new
        // adaptive scenecut code.
        const BIAS: f64 = 0.7;
        let threshold = intra_cost * (1.0 - BIAS);

        let mut result = ScenecutResult::new(
            mv_inter_cost.mean,
            imp_block_diff.mean,
            self.importance_threshold(imp_block_diff.avg_luma_8bit),
            threshold,
            imp_block_diff.avg_luma_8bit,
        );
        result.active_block_fraction = active_region.fraction();
        result.active_crop = (!active_region.is_full()).then(|| {
            [
                active_region.top,
                active_region.rows - active_region.bottom,
                active_region.left,
                active_region.cols - active_region.right,
            ]
        });

        // Good-block gate input: count a well-matched block as "tracking"
        // evidence only when it carries detail (intra cost above the floor);
        // flat blocks match across any cut and prove nothing.
        let min_intra_ratio = self.tuning.importance_cut_good_block_min_intra_ratio;
        let inter_block_costs = inter_block_costs.expect("inter block costs should be set");
        result.textured_good_block_ratio = if min_intra_ratio > 0.0
            && active_region.count() > 0
            && inter_block_costs.len() == intra_block_costs.len()
        {
            let intra_floor = intra_cost * min_intra_ratio;
            let good_threshold = mv_inter_cost.mean * 0.25;
            let textured_good = active_region
                .indices()
                .filter(|&idx| {
                    inter_block_costs[idx] <= good_threshold
                        && f64::from(intra_block_costs[idx]) >= intra_floor
                })
                .count();
            textured_good as f64 / active_region.count() as f64
        } else {
            // Read the estimate, not `result`: the static ratios are copied
            // onto `result` further down, so `result.static_good_block_ratio`
            // is still the constructor default here.
            mv_inter_cost.static_good_block_ratio
        };
        result.dc_free_good_block_ratio = mv_inter_cost.dc_free_good_block_ratio;
        result.me_dc_free_good_block_ratio = mv_inter_cost.me_dc_free_good_block_ratio;
        result.me_dc_free_cost = mv_inter_cost.me_dc_free_mean;
        result.structure_match_ratio = mv_inter_cost.structure_match_ratio;
        result.structure_coverage = mv_inter_cost.structure_coverage;
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
            let active_rect = (!active_region.is_full()).then(|| {
                (
                    active_region.left * IMPORTANCE_BLOCK_SIZE,
                    active_region.right * IMPORTANCE_BLOCK_SIZE,
                    active_region.top * IMPORTANCE_BLOCK_SIZE,
                    active_region.bottom * IMPORTANCE_BLOCK_SIZE,
                )
            });
            Some(frame_luma_signature_8bit(frame2, self.bit_depth, active_rect))
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
