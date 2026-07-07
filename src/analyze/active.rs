//! Letterbox/pillarbox detection on the importance-block grid.
//!
//! Frames with black bars (letterboxing, or mid-video aspect-ratio switches)
//! dilute every per-block metric: bar blocks contribute near-zero inter/intra
//! cost and count as "static good" blocks, which both suppresses cost ratios
//! and inflates `static_good_block_ratio` by exactly the bar fraction of the
//! frame. Detecting the active picture region and excluding bar blocks from
//! metric aggregation keeps scores comparable between bared and bar-less
//! content without any per-source tuning.

use crate::analyze::importance::IMPORTANCE_BLOCK_SIZE;

/// Mean 8-bit luma at or below which a block is considered "black" for bar
/// detection. Limited-range black is 16; a small margin absorbs encoder noise
/// and dither without swallowing legitimately dark picture content.
const BAR_BLACK_LEVEL_8BIT: u64 = 18;

/// Per-row/column fraction of blocks allowed to exceed the black level while
/// the row/column still counts as a bar (tolerates burned-in logos and noise
/// specks inside the bars).
const BAR_NONBLACK_TOLERANCE: f64 = 0.03;

/// If stripping bars would leave less than this fraction of rows or columns,
/// the frame is treated as having no bars at all. Guards fades to black and
/// dark credit rolls from being mistaken for gigantic bars.
const MIN_ACTIVE_FRACTION_PER_AXIS: f64 = 0.5;

/// Active picture region on the importance-block grid, in block coordinates.
/// `top..bottom` x `left..right` are the non-bar blocks; a full-frame region
/// means no bars were detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveRegion {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
    pub cols: usize,
    pub rows: usize,
}

impl ActiveRegion {
    pub const fn full(cols: usize, rows: usize) -> Self {
        Self {
            top: 0,
            bottom: rows,
            left: 0,
            right: cols,
            cols,
            rows,
        }
    }

    pub const fn is_full(&self) -> bool {
        self.top == 0 && self.left == 0 && self.bottom == self.rows && self.right == self.cols
    }

    pub const fn count(&self) -> usize {
        (self.bottom - self.top) * (self.right - self.left)
    }

    pub fn fraction(&self) -> f64 {
        if self.cols * self.rows == 0 {
            1.0
        } else {
            self.count() as f64 / (self.cols * self.rows) as f64
        }
    }

    /// Iterates flat block indices (row-major, matching every block-cost
    /// vector in the analysis path) inside the active region.
    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        let cols = self.cols;
        let (left, right) = (self.left, self.right);
        (self.top..self.bottom).flat_map(move |y| (left..right).map(move |x| y * cols + x))
    }
}

/// Detects the active (non-bar) region from per-block luma sums of the two
/// compared frames.
///
/// A block is "black" only when it is black in *both* frames: bars must be
/// present on each side of the comparison to be excluded, so a cut into or out
/// of bared content still sees the bar area as changed content. Bars are
/// stripped as contiguous runs of black rows from the top/bottom (letterbox)
/// and black columns from the left/right (pillarbox, evaluated over the
/// remaining rows).
pub(crate) fn detect_active_region(
    org_sums: &[u64],
    ref_sums: &[u64],
    cols: usize,
    rows: usize,
    bit_depth: usize,
) -> ActiveRegion {
    let full = ActiveRegion::full(cols, rows);
    if cols == 0 || rows == 0 {
        return full;
    }
    debug_assert_eq!(org_sums.len(), cols * rows);
    debug_assert_eq!(ref_sums.len(), cols * rows);

    let block_pixels = (IMPORTANCE_BLOCK_SIZE * IMPORTANCE_BLOCK_SIZE) as u64;
    let black_sum_max = BAR_BLACK_LEVEL_8BIT * block_pixels << (bit_depth - 8);
    let is_black =
        |idx: usize| org_sums[idx] <= black_sum_max && ref_sums[idx] <= black_sum_max;

    let max_nonblack_in_row = (cols as f64 * BAR_NONBLACK_TOLERANCE) as usize;
    let row_is_bar = |y: usize| {
        (0..cols).filter(|&x| !is_black(y * cols + x)).count() <= max_nonblack_in_row
    };

    let mut top = 0;
    while top < rows && row_is_bar(top) {
        top += 1;
    }
    let mut bottom = rows;
    while bottom > top && row_is_bar(bottom - 1) {
        bottom -= 1;
    }
    if ((bottom - top) as f64) < rows as f64 * MIN_ACTIVE_FRACTION_PER_AXIS {
        return full;
    }

    let max_nonblack_in_col = ((bottom - top) as f64 * BAR_NONBLACK_TOLERANCE) as usize;
    let col_is_bar = |x: usize| {
        (top..bottom).filter(|&y| !is_black(y * cols + x)).count() <= max_nonblack_in_col
    };

    let mut left = 0;
    while left < cols && col_is_bar(left) {
        left += 1;
    }
    let mut right = cols;
    while right > left && col_is_bar(right - 1) {
        right -= 1;
    }
    if ((right - left) as f64) < cols as f64 * MIN_ACTIVE_FRACTION_PER_AXIS {
        left = 0;
        right = cols;
    }

    ActiveRegion {
        top,
        bottom,
        left,
        right,
        cols,
        rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sums(cols: usize, rows: usize, fill: u64, bar_rows: usize) -> Vec<u64> {
        let block_pixels = (IMPORTANCE_BLOCK_SIZE * IMPORTANCE_BLOCK_SIZE) as u64;
        let mut v = vec![fill * block_pixels; cols * rows];
        for y in (0..bar_rows).chain(rows - bar_rows..rows) {
            for x in 0..cols {
                v[y * cols + x] = 16 * block_pixels;
            }
        }
        v
    }

    #[test]
    fn no_bars_detects_full() {
        let s = sums(16, 9, 60, 0);
        let region = detect_active_region(&s, &s, 16, 9, 8);
        assert!(region.is_full());
        assert!((region.fraction() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn letterbox_stripped_both_sides() {
        let s = sums(16, 10, 60, 2);
        let region = detect_active_region(&s, &s, 16, 10, 8);
        assert_eq!((region.top, region.bottom), (2, 8));
        assert_eq!((region.left, region.right), (0, 16));
        assert_eq!(region.count(), 16 * 6);
    }

    #[test]
    fn bar_only_in_one_frame_is_not_stripped() {
        let bared = sums(16, 10, 60, 2);
        let clean = sums(16, 10, 60, 0);
        let region = detect_active_region(&bared, &clean, 16, 10, 8);
        assert!(region.is_full());
    }

    #[test]
    fn near_black_frame_falls_back_to_full() {
        let s = sums(16, 10, 60, 5);
        let region = detect_active_region(&s, &s, 16, 10, 8);
        assert!(region.is_full());
    }

    #[test]
    fn high_bit_depth_scales_black_level() {
        let block_pixels = (IMPORTANCE_BLOCK_SIZE * IMPORTANCE_BLOCK_SIZE) as u64;
        let mut s = vec![240u64 * block_pixels; 16 * 10];
        for y in [0usize, 1, 8, 9] {
            for x in 0..16 {
                s[y * 16 + x] = 64 * block_pixels; // 10-bit limited black
            }
        }
        let region = detect_active_region(&s, &s, 16, 10, 10);
        assert_eq!((region.top, region.bottom), (2, 8));
    }

    #[test]
    fn indices_are_row_major_within_region() {
        let s = sums(4, 4, 60, 1);
        let region = detect_active_region(&s, &s, 4, 4, 8);
        assert_eq!((region.top, region.bottom), (1, 3));
        let idx: Vec<_> = region.indices().collect();
        assert_eq!(idx, vec![4, 5, 6, 7, 8, 9, 10, 11]);
    }
}
