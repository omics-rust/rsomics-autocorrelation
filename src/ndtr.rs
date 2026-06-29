//! Inverse standard-normal CDF `ndtri`, ported read-only for value-exactness
//! with scipy.
//!
//! `scipy.stats.norm.ppf(1 - alpha/2)` is the multiplier in the Bartlett-formula
//! and 1/√n confidence intervals. This is a verbatim transcription of cephes
//! `ndtri.c` (public domain, vendored in SciPy `scipy/special/cephes/`, BSD-3);
//! only the control flow is rewritten in Rust.

#![allow(clippy::excessive_precision)]
#![allow(clippy::manual_range_contains)]

fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

fn p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

const NDTRI_S2PI: f64 = 2.50662827463100050242e0;

const NDTRI_P0: [f64; 5] = [
    -5.99633501014107895267e1,
    9.80010754185999661536e1,
    -5.66762857469070293439e1,
    1.39312609387279679503e1,
    -1.23916583867381258016e0,
];
const NDTRI_Q0: [f64; 8] = [
    1.95448858338141759834e0,
    4.67627912898881538453e0,
    8.63602421390890590575e1,
    -2.25462687854119370527e2,
    2.00260212380060660359e2,
    -8.20372256168333339912e1,
    1.59056225126211695515e1,
    -1.18331621121330003142e0,
];
const NDTRI_P1: [f64; 9] = [
    4.05544892305962419923e0,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539e0,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4,
];
const NDTRI_Q1: [f64; 8] = [
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979e0,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4,
];
const NDTRI_P2: [f64; 9] = [
    3.23774891776946035970e0,
    6.91522889068984211695e0,
    3.93881025292474443415e0,
    1.33303460815807542389e0,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9,
];
const NDTRI_Q2: [f64; 8] = [
    6.02427039364742014255e0,
    3.67983563856160859403e0,
    1.37702099489081330271e0,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9,
];

/// Inverse standard-normal CDF: the `x` with Φ(x) = `y0`. cephes `ndtri`.
pub fn ndtri(y0: f64) -> f64 {
    if y0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 == 1.0 {
        return f64::INFINITY;
    }
    if y0 < 0.0 || y0 > 1.0 {
        return f64::NAN;
    }

    let mut code = true;
    let mut y = y0;
    let exp_m2 = 0.13533528323661269189;
    if y > 1.0 - exp_m2 {
        y = 1.0 - y;
        code = false;
    }
    if y > exp_m2 {
        y -= 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * polevl(y2, &NDTRI_P0) / p1evl(y2, &NDTRI_Q0));
        return x * NDTRI_S2PI;
    }

    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * polevl(z, &NDTRI_P1) / p1evl(z, &NDTRI_Q1)
    } else {
        z * polevl(z, &NDTRI_P2) / p1evl(z, &NDTRI_Q2)
    };
    let xr = x0 - x1;
    if code { -xr } else { xr }
}

#[cfg(test)]
mod tests {
    use super::ndtri;

    fn rel(got: f64, want: f64) -> f64 {
        (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn ndtri_known() {
        assert!(rel(ndtri(0.975), 1.959963984540054) < 1e-13);
        assert!(rel(ndtri(0.95), 1.6448536269514722) < 1e-13);
        assert_eq!(ndtri(0.5), 0.0);
    }
}
