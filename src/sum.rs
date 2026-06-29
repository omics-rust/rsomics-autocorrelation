//! numpy's pairwise summation, reproduced bit-for-bit.
//!
//! numpy reduces with `pairwise_sum`: leaf blocks of up to 128 elements use
//! eight interleaved accumulators, larger spans recurse on a half split rounded
//! down to a multiple of eight. The demeaning (`x - x.mean()`) and the lag
//! dot-products both pass through this order, so the autocovariances stay within
//! a few ULP of statsmodels — a naive left-fold drifts in the low bits at large
//! N. The lag products `xo[k:]·xo[:-k]` go through BLAS `ddot` in statsmodels,
//! whose SIMD reduction is not portably reproducible; pairwise summation is the
//! closest portable order and the residual sits at ~1e-15 relative.

const BLOCK: usize = 128;

pub fn pairwise_sum(data: &[f64]) -> f64 {
    sum_range(data, 0, data.len())
}

/// Pairwise summation of `a[i] * b[i]`, matching numpy's reduction of the
/// elementwise product without materialising it.
pub fn pairwise_dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    dot_range(a, b, 0, a.len())
}

fn sum_range(data: &[f64], lo: usize, n: usize) -> f64 {
    if n < 8 {
        let mut s = 0.0;
        for &v in &data[lo..lo + n] {
            s += v;
        }
        s
    } else if n <= BLOCK {
        let mut acc = [
            data[lo],
            data[lo + 1],
            data[lo + 2],
            data[lo + 3],
            data[lo + 4],
            data[lo + 5],
            data[lo + 6],
            data[lo + 7],
        ];
        let mut i = 8;
        while i + 8 <= n {
            for j in 0..8 {
                acc[j] += data[lo + i + j];
            }
            i += 8;
        }
        let mut res =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        while i < n {
            res += data[lo + i];
            i += 1;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        sum_range(data, lo, half) + sum_range(data, lo + half, n - half)
    }
}

fn dot_range(a: &[f64], b: &[f64], lo: usize, n: usize) -> f64 {
    if n < 8 {
        let mut s = 0.0;
        for i in lo..lo + n {
            s += a[i] * b[i];
        }
        s
    } else if n <= BLOCK {
        let mut acc = [
            a[lo] * b[lo],
            a[lo + 1] * b[lo + 1],
            a[lo + 2] * b[lo + 2],
            a[lo + 3] * b[lo + 3],
            a[lo + 4] * b[lo + 4],
            a[lo + 5] * b[lo + 5],
            a[lo + 6] * b[lo + 6],
            a[lo + 7] * b[lo + 7],
        ];
        let mut i = 8;
        while i + 8 <= n {
            for j in 0..8 {
                acc[j] += a[lo + i + j] * b[lo + i + j];
            }
            i += 8;
        }
        let mut res =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        while i < n {
            res += a[lo + i] * b[lo + i];
            i += 1;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        dot_range(a, b, lo, half) + dot_range(a, b, lo + half, n - half)
    }
}

#[cfg(test)]
mod tests {
    use super::{pairwise_dot, pairwise_sum};

    #[test]
    fn small_naive_sum() {
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn block_boundary() {
        let v: Vec<f64> = (1..=200).map(f64::from).collect();
        assert_eq!(pairwise_sum(&v), (200.0 * 201.0) / 2.0);
    }

    #[test]
    fn dot_matches_self_sum_of_squares() {
        let v: Vec<f64> = (1..=200).map(f64::from).collect();
        let sq: Vec<f64> = v.iter().map(|x| x * x).collect();
        assert_eq!(pairwise_dot(&v, &v), pairwise_sum(&sq));
    }
}
