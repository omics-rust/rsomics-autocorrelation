//! Autocovariance, autocorrelation, Ljung-Box / Box-Pierce Q, and Bartlett
//! confidence intervals — `statsmodels.tsa.stattools.{acovf,acf,q_stat}`.

use crate::ndtr::ndtri;
use crate::sum::{pairwise_dot, pairwise_sum};

/// Direct (non-FFT) autocovariance at lags `0..=nlags`, demeaned.
///
/// Matches `statsmodels.tsa.stattools.acovf(x, adjusted, demean=True,
/// fft=False, nlag=nlags)`: `acov[0] = xo·xo`, `acov[k] = xo[k:]·xo[:-k]`,
/// divided by `n` (biased) or `n-k` (`adjusted`). statsmodels routes the dot
/// products through BLAS; we use numpy's pairwise order, which agrees to a few
/// ULP (the lag products are not bit-portable across BLAS implementations).
pub fn acovf(x: &[f64], nlags: usize, adjusted: bool) -> Vec<f64> {
    let n = x.len();
    let mean = pairwise_sum(x) / n as f64;
    let xo: Vec<f64> = x.iter().map(|v| v - mean).collect();

    let mut acov = vec![0.0; nlags + 1];
    acov[0] = pairwise_dot(&xo, &xo);
    for k in 1..=nlags {
        acov[k] = pairwise_dot(&xo[k..], &xo[..n - k]);
    }
    for (k, a) in acov.iter_mut().enumerate() {
        *a /= if adjusted { (n - k) as f64 } else { n as f64 };
    }
    acov
}

/// Autocorrelation at lags `0..=nlags`: `acovf[k] / acovf[0]`.
pub fn acf(x: &[f64], nlags: usize, adjusted: bool) -> Vec<f64> {
    let avf = acovf(x, nlags, adjusted);
    let a0 = avf[0];
    avf.iter().map(|a| a / a0).collect()
}

/// statsmodels default `nlags` for `acf`: `min(int(10*log10(n)), n-1)`.
pub fn default_acf_nlags(n: usize) -> usize {
    ((10.0 * (n as f64).log10()) as usize).min(n - 1)
}

/// Ljung-Box (or Box-Pierce) Q statistics with chi-square p-values for
/// `acf[1..=k]`, `k = 1..=nlags`. Mirrors `q_stat` / `acorr_ljungbox`:
/// Ljung-Box `Q_k = n(n+2)·Σ acf[j]²/(n-j)`, Box-Pierce `Q_k = n·Σ acf[j]²`,
/// `p = chi2.sf(Q_k, k - model_df)` (NaN when `k - model_df <= 0`).
pub fn q_stat(
    acf_tail: &[f64],
    n: usize,
    boxpierce: bool,
    model_df: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nf = n as f64;
    let scale = if boxpierce { nf } else { nf * (nf + 2.0) };

    let mut q = Vec::with_capacity(acf_tail.len());
    let mut p = Vec::with_capacity(acf_tail.len());
    let mut csum = 0.0;
    for (i, &r) in acf_tail.iter().enumerate() {
        let lag = i + 1;
        let term = if boxpierce {
            r * r
        } else {
            r * r / (nf - lag as f64)
        };
        csum += term;
        let qk = scale * csum;
        q.push(qk);

        let dof = lag as isize - model_df as isize;
        if dof > 0 {
            p.push(chi2_sf(qk, dof as f64));
        } else {
            p.push(f64::NAN);
        }
    }
    (q, p)
}

/// Bartlett-formula confidence half-widths for `acf` at lags `0..=nlags`,
/// matching `acf(..., alpha=...)`: `var[0]=0`, `var[1]=1/n`,
/// `var[k>=2] = (1/n)(1 + 2·Σ_{j<k} acf[j]²)`, half-width
/// `ndtri(1-alpha/2)·sqrt(var)`.
pub fn bartlett_halfwidth(acf: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let z = ndtri(1.0 - alpha / 2.0);
    let nf = n as f64;
    let m = acf.len();
    let mut var = vec![1.0 / nf; m];
    if m > 0 {
        var[0] = 0.0;
    }
    if m > 1 {
        var[1] = 1.0 / nf;
    }
    let mut csum = 0.0;
    for k in 2..m {
        csum += acf[k - 1] * acf[k - 1];
        var[k] = (1.0 / nf) * (1.0 + 2.0 * csum);
    }
    var.iter().map(|v| z * v.sqrt()).collect()
}

/// Survival function of the chi-square distribution: `chi2.sf(x, df)` =
/// `igamc(df/2, x/2)`.
fn chi2_sf(x: f64, df: f64) -> f64 {
    crate::igamc::igamc(df / 2.0, x / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acf_lag0_is_one() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let r = acf(&x, 3, false);
        assert_eq!(r[0], 1.0);
    }

    #[test]
    fn default_nlags_rule() {
        assert_eq!(default_acf_nlags(200), 23);
        assert_eq!(default_acf_nlags(5), 4);
    }

    #[test]
    fn q_stat_p_in_unit_interval() {
        let acf_tail = [0.5, 0.2, -0.1];
        let (q, p) = q_stat(&acf_tail, 100, false, 0);
        assert_eq!(q.len(), 3);
        for pv in p {
            assert!((0.0..=1.0).contains(&pv));
        }
    }
}
