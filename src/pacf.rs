//! Partial autocorrelation — `statsmodels.tsa.stattools.pacf`.
//!
//! statsmodels' default `method="ywadjusted"` solves the order-k Yule-Walker
//! system at each lag and takes the last coefficient. That system is solved by
//! the Durbin-Levinson recursion here, whose diagonal is exactly that sequence
//! of last coefficients; statsmodels reaches the same value through a LAPACK
//! `solve`, so the two agree to a few ULP rather than bit-for-bit. The `ld` /
//! `ldb` methods *are* the Levinson-Durbin recursion in statsmodels and match
//! the recursion here exactly (given the same autocovariances).

use crate::acf::acovf;
use crate::ndtr::ndtri;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Yule-Walker with sample-size adjustment (statsmodels default `ywadjusted`).
    Yw,
    /// Yule-Walker, maximum-likelihood / no adjustment (`ywm`).
    Ywm,
    /// Levinson-Durbin with bias correction (`ld`).
    Ld,
    /// Levinson-Durbin without bias correction (`ldb`).
    Ldb,
    /// Ordinary least squares on lags plus constant (`ols`, efficient variant).
    Ols,
}

impl Method {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "yw" | "ywa" | "ywadjusted" | "yw_adjusted" => Some(Method::Yw),
            "ywm" | "ywmle" | "yw_mle" => Some(Method::Ywm),
            "ld" | "lda" | "ldadjusted" | "ld_adjusted" => Some(Method::Ld),
            "ldb" | "ldbiased" | "ld_biased" => Some(Method::Ldb),
            "ols" => Some(Method::Ols),
            _ => None,
        }
    }
}

/// statsmodels default `nlags` for `pacf`: `max(min(int(10*log10(n)), n//2-1), 1)`.
pub fn default_pacf_nlags(n: usize) -> usize {
    let raw = ((10.0 * (n as f64).log10()) as usize).min(n / 2 - 1);
    raw.max(1)
}

/// Partial autocorrelation at lags `0..=nlags` for the chosen method.
pub fn pacf(x: &[f64], nlags: usize, method: Method) -> Vec<f64> {
    match method {
        Method::Yw | Method::Ld => levinson_durbin_pacf(&acovf(x, nlags, true), nlags),
        Method::Ywm | Method::Ldb => levinson_durbin_pacf(&acovf(x, nlags, false), nlags),
        Method::Ols => pacf_ols(x, nlags),
    }
}

/// Levinson-Durbin recursion returning the partial autocorrelations (the
/// diagonal of `phi`), `pacf[0] = 1`. Faithful to statsmodels `levinson_durbin`.
fn levinson_durbin_pacf(sxx: &[f64], order: usize) -> Vec<f64> {
    let mut phi = vec![vec![0.0; order + 1]; order + 1];
    let mut sig = vec![0.0; order + 1];

    phi[1][1] = sxx[1] / sxx[0];
    sig[1] = sxx[0] - phi[1][1] * sxx[1];
    for k in 2..=order {
        let mut acc = 0.0;
        for i in 1..k {
            acc += phi[i][k - 1] * sxx[k - i];
        }
        phi[k][k] = (sxx[k] - acc) / sig[k - 1];
        for j in 1..k {
            phi[j][k] = phi[j][k - 1] - phi[k][k] * phi[k - j][k - 1];
        }
        sig[k] = sig[k - 1] * (1.0 - phi[k][k] * phi[k][k]);
    }

    let mut pacf = vec![0.0; order + 1];
    pacf[0] = 1.0;
    for k in 1..=order {
        pacf[k] = phi[k][k];
    }
    pacf
}

/// OLS partial autocorrelation, efficient variant: at each lag `k` regress the
/// series on its first `k` lags plus a constant (re-estimating the mean), over
/// the maximal sample, and take the highest-lag coefficient. Matches
/// `pacf_ols(x, efficient=True)`. The normal-equations solve agrees with
/// statsmodels' `lstsq` (SVD) to ~1e-10 rather than bit-for-bit.
fn pacf_ols(x: &[f64], nlags: usize) -> Vec<f64> {
    let mut pacf = vec![0.0; nlags + 1];
    pacf[0] = 1.0;
    let n = x.len();
    for k in 1..=nlags {
        // Rows k..n of the design [1, x_{t-1}, ..., x_{t-k}] predicting x_t.
        let rows = n - k;
        let cols = k + 1;
        let mut design = vec![vec![0.0; cols]; rows];
        let mut y = vec![0.0; rows];
        for (r, t) in (k..n).enumerate() {
            design[r][0] = 1.0;
            for j in 1..=k {
                design[r][j] = x[t - j];
            }
            y[r] = x[t];
        }
        let beta = ols_solve(&design, &y);
        pacf[k] = beta[k];
    }
    pacf
}

/// Solve `min ||A·b - y||` via the normal equations `(AᵀA) b = Aᵀy` with
/// Gaussian elimination + partial pivoting.
fn ols_solve(a: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let cols = a[0].len();
    let mut ata = vec![vec![0.0; cols]; cols];
    let mut aty = vec![0.0; cols];
    for (row, &yi) in a.iter().zip(y) {
        for i in 0..cols {
            aty[i] += row[i] * yi;
            for j in 0..cols {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    solve_linear(&mut ata, &mut aty);
    aty
}

/// Gaussian elimination with partial pivoting, solving `a·x = b` in place;
/// the solution is left in `b`.
fn solve_linear(a: &mut [Vec<f64>], b: &mut [f64]) {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let pivot_row = a[col].clone();
        let bcol = b[col];
        let diag = pivot_row[col];
        for r in col + 1..n {
            let f = a[r][col] / diag;
            for (ac, &pc) in a[r].iter_mut().zip(&pivot_row).skip(col) {
                *ac -= f * pc;
            }
            b[r] -= f * bcol;
        }
    }
    for col in (0..n).rev() {
        let mut s = b[col];
        for (c, &acc) in a[col].iter().enumerate().skip(col + 1) {
            s -= acc * b[c];
        }
        b[col] = s / a[col][col];
    }
}

/// Confidence half-width for pacf at lags `>= 1`: `ndtri(1-alpha/2)/sqrt(n)`
/// (statsmodels uses `varacf = 1/n` for all lags; lag 0 is exact).
pub fn pacf_halfwidth(n: usize, alpha: f64) -> f64 {
    ndtri(1.0 - alpha / 2.0) * (1.0 / n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacf_lag0_is_one() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        for m in [
            Method::Yw,
            Method::Ywm,
            Method::Ld,
            Method::Ldb,
            Method::Ols,
        ] {
            let p = pacf(&x, 5, m);
            assert_eq!(p[0], 1.0, "{m:?}");
        }
    }

    #[test]
    fn method_aliases() {
        assert_eq!(Method::parse("ywadjusted"), Some(Method::Yw));
        assert_eq!(Method::parse("ld"), Some(Method::Ld));
        assert_eq!(Method::parse("nope"), None);
    }

    #[test]
    fn default_nlags_rule() {
        assert_eq!(default_pacf_nlags(200), 23);
    }
}
