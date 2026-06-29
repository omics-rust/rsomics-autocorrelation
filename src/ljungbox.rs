//! Ljung-Box / Box-Pierce test — `statsmodels.stats.diagnostic.acorr_ljungbox`.

use crate::acf::{acf, q_stat};

pub struct LjungBox {
    pub lags: Vec<usize>,
    pub lb_stat: Vec<f64>,
    pub lb_pvalue: Vec<f64>,
    /// Present when Box-Pierce was requested.
    pub bp_stat: Option<Vec<f64>>,
    pub bp_pvalue: Option<Vec<f64>>,
}

/// statsmodels default `lags` for `acorr_ljungbox`: `min(10, n//5)`.
pub fn default_ljungbox_lags(n: usize) -> usize {
    10.min(n / 5)
}

/// Ljung-Box (and optionally Box-Pierce) statistics for lags `1..=maxlag`,
/// `Q_k = n(n+2)·Σ acf[j]²/(n-j)` and `Q_k = n·Σ acf[j]²` respectively, with
/// `p = chi2.sf(Q_k, k - model_df)`. The autocorrelations come from
/// `acf(x, nlags=maxlag, fft=False)` (biased), matching `acorr_ljungbox`.
pub fn acorr_ljungbox(x: &[f64], maxlag: usize, boxpierce: bool, model_df: usize) -> LjungBox {
    let n = x.len();
    let sacf = acf(x, maxlag, false);
    let tail = &sacf[1..=maxlag];

    let (lb_stat, lb_pvalue) = q_stat(tail, n, false, model_df);
    let lags: Vec<usize> = (1..=maxlag).collect();

    let (bp_stat, bp_pvalue) = if boxpierce {
        let (q, p) = q_stat(tail, n, true, model_df);
        (Some(q), Some(p))
    } else {
        (None, None)
    };

    LjungBox {
        lags,
        lb_stat,
        lb_pvalue,
        bp_stat,
        bp_pvalue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_lags_rule() {
        assert_eq!(default_ljungbox_lags(200), 10);
        assert_eq!(default_ljungbox_lags(30), 6);
    }

    #[test]
    fn shapes_and_ranges() {
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.2).sin()).collect();
        let r = acorr_ljungbox(&x, 10, true, 0);
        assert_eq!(r.lags.len(), 10);
        assert_eq!(r.lb_stat.len(), 10);
        assert!(r.bp_stat.as_ref().unwrap().len() == 10);
        for &p in &r.lb_pvalue {
            assert!((0.0..=1.0).contains(&p) || p.is_nan());
        }
    }
}
