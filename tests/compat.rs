//! Value-exact compatibility with statsmodels 0.14.6 (`tsa.stattools` and
//! `stats.diagnostic`).
//!
//! Golden values were produced once by running statsmodels on LCG-generated
//! series; this test re-derives the series in Rust and asserts the results
//! match with no statsmodels present. statsmodels' direct (`fft=False`)
//! autocovariance routes its lag dot-products through BLAS, whose SIMD reduction
//! order is not portably reproducible — so the agreement is a few ULP (~1e-15
//! relative) rather than bit-for-bit. The PACF Durbin-Levinson recursion is
//! bit-faithful given the same autocovariances. We assert every numeric column
//! (acf, pacf, Q, p-values, confidence bounds) to 1e-12 relative; the goldens
//! are stored as IEEE-754 hex so serde_json's float parser cannot mask the
//! comparison.

use rsomics_autocorrelation::{
    acf::{acovf, bartlett_halfwidth, q_stat},
    acorr_ljungbox,
    pacf::{Method, pacf},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    kind: String,
    series: String,
    n: usize,
    #[serde(default)]
    nlags: usize,
    #[serde(default)]
    maxlag: usize,
    #[serde(default)]
    adjusted: bool,
    #[serde(default)]
    method: String,
    #[serde(default)]
    boxpierce: bool,
    #[serde(default)]
    acf_bits: Vec<String>,
    #[serde(default)]
    lower_bits: Vec<String>,
    #[serde(default)]
    upper_bits: Vec<String>,
    #[serde(default)]
    qstat_bits: Vec<String>,
    #[serde(default)]
    qpvalue_bits: Vec<String>,
    #[serde(default)]
    pacf_bits: Vec<String>,
    #[serde(default)]
    lb_stat_bits: Vec<String>,
    #[serde(default)]
    lb_pvalue_bits: Vec<String>,
    #[serde(default)]
    bp_stat_bits: Vec<String>,
    #[serde(default)]
    bp_pvalue_bits: Vec<String>,
}

#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

fn bits(hex: &str) -> f64 {
    let mut le = [0u8; 8];
    let raw = hex::decode(hex);
    le.copy_from_slice(&raw);
    f64::from_le_bytes(le)
}

mod hex {
    pub fn decode(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }
}

const MASK: u64 = u64::MAX;

fn ar1(n: usize) -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut prev = 0.0;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                & MASK;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            prev = 0.7 * prev + (u - 0.5);
            prev
        })
        .collect()
}

fn white(n: usize) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                & MASK;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            u * 2.0 - 1.0
        })
        .collect()
}

fn series(name: &str) -> Vec<f64> {
    match name {
        "ar1_50" => ar1(50),
        "ar1_5000" => ar1(5000),
        "ar1_1e6" => ar1(1_000_000),
        "white_2000" => white(2000),
        other => panic!("unknown series {other}"),
    }
}

/// Relative-error closeness; statsmodels' BLAS-routed acovf is not bit-portable,
/// so a few ULP of slack is expected and bounded here.
fn close(got: f64, want: f64, tol: f64, ctx: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{ctx}: got {got}, want NaN");
        return;
    }
    let rel = (got - want).abs() / want.abs().max(1.0);
    assert!(
        rel <= tol,
        "{ctx}: got {got}, want {want}, rel {rel:.3e} > {tol:.0e}"
    );
}

const TOL: f64 = 1e-12;

#[test]
fn matches_statsmodels_golden() {
    let golden: Golden = serde_json::from_slice(include_bytes!("golden/expected.json")).unwrap();

    for c in &golden.cases {
        let x = series(&c.series);
        let tag = format!("{} {} n={}", c.kind, c.series, c.n);

        match c.kind.as_str() {
            "acf" => {
                let avf = acovf(&x, c.nlags, c.adjusted);
                let a0 = avf[0];
                let acf: Vec<f64> = avf.iter().map(|a| a / a0).collect();
                let half = bartlett_halfwidth(&acf, c.n, 0.05);
                let (q, p) = q_stat(&acf[1..], c.n, false, 0);

                for (k, w) in c.acf_bits.iter().enumerate() {
                    close(acf[k], bits(w), TOL, &format!("{tag} acf[{k}]"));
                }
                for (k, w) in c.lower_bits.iter().enumerate() {
                    close(acf[k] - half[k], bits(w), TOL, &format!("{tag} lower[{k}]"));
                }
                for (k, w) in c.upper_bits.iter().enumerate() {
                    close(acf[k] + half[k], bits(w), TOL, &format!("{tag} upper[{k}]"));
                }
                for (i, w) in c.qstat_bits.iter().enumerate() {
                    close(q[i], bits(w), TOL, &format!("{tag} qstat[{i}]"));
                }
                for (i, w) in c.qpvalue_bits.iter().enumerate() {
                    close(p[i], bits(w), TOL, &format!("{tag} qpvalue[{i}]"));
                }
            }
            "pacf" => {
                let method = Method::parse(&c.method).unwrap();
                let p = pacf(&x, c.nlags, method);
                for (k, w) in c.pacf_bits.iter().enumerate() {
                    close(
                        p[k],
                        bits(w),
                        TOL,
                        &format!("{tag} pacf[{k}] method={}", c.method),
                    );
                }
            }
            "ljungbox" => {
                let r = acorr_ljungbox(&x, c.maxlag, c.boxpierce, 0);
                for (i, w) in c.lb_stat_bits.iter().enumerate() {
                    close(r.lb_stat[i], bits(w), TOL, &format!("{tag} lb_stat[{i}]"));
                }
                for (i, w) in c.lb_pvalue_bits.iter().enumerate() {
                    close(
                        r.lb_pvalue[i],
                        bits(w),
                        TOL,
                        &format!("{tag} lb_pvalue[{i}]"),
                    );
                }
                if c.boxpierce {
                    let bp = r.bp_stat.as_ref().unwrap();
                    let bpp = r.bp_pvalue.as_ref().unwrap();
                    for (i, w) in c.bp_stat_bits.iter().enumerate() {
                        close(bp[i], bits(w), TOL, &format!("{tag} bp_stat[{i}]"));
                    }
                    for (i, w) in c.bp_pvalue_bits.iter().enumerate() {
                        close(bpp[i], bits(w), TOL, &format!("{tag} bp_pvalue[{i}]"));
                    }
                }
            }
            other => panic!("unknown kind {other}"),
        }
    }
}
