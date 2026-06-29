//! Regularized upper incomplete gamma `igamc(a, x)`, ported read-only for
//! value-exactness with scipy.
//!
//! `chi2.sf(x, df) = igamc(df/2, x/2)` drives the Ljung-Box / Box-Pierce
//! p-values. This is a faithful transcription of the cephes C sources scipy is
//! itself derived from — `gamma.c`, `igam.c`, `unity.c`, `ndtr.c` (the `erfc`
//! used by the asymptotic series). Algorithms and coefficients are copied
//! verbatim; only the control flow is rewritten in Rust.
//!
//! Source: Stephen L. Moshier's Cephes Math Library (public domain), as
//! vendored in SciPy (`scipy/special/cephes/`, BSD-3). Cited in the crate
//! README `## Origin`.

// Verbatim transcription of cephes C; idioms that read as clippy lints
// (shared asymptotic-regime guards, late init) preserve the line-by-line
// correspondence with the reference source and are allowed module-wide.
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]

const MACHEP: f64 = 1.11022302462515654042e-16;
const MAXLOG: f64 = 7.09782712893383996732e2;
const MAXITER: usize = 2000;
const EULER: f64 = 0.577215664901532860606512090082402431;

const ZETA: [f64; 40] = [
    1.6449340668482266,
    1.202056903159594,
    1.0823232337111381,
    1.0369277551433704,
    1.0173430619844488,
    1.008349277381923,
    1.0040773561979446,
    1.0020083928260826,
    1.0009945751278182,
    1.0004941886041194,
    1.0002460865533078,
    1.0001227133475785,
    1.0000612481350586,
    1.0000305882363072,
    1.0000152822594084,
    1.0000076371976376,
    1.000003817293265,
    1.0000019082127163,
    1.0000009539620338,
    1.0000004769329867,
    1.0000002384505027,
    1.000000119219926,
    1.000000059608189,
    1.0000000298035034,
    1.0000000149015549,
    1.0000000074507118,
    1.000000003725334,
    1.0000000018626598,
    1.0000000009313275,
    1.0000000004656628,
    1.000000000232831,
    1.0000000001164155,
    1.0000000000582077,
    1.0000000000291038,
    1.000000000014552,
    1.000000000007276,
    1.000000000003638,
    1.000000000001819,
    1.0000000000009095,
    1.0000000000004547,
];

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

// Lanczos approximation (lanczos.h, Boost-derived).

const LANCZOS_G: f64 = 6.024680040776729583740234375;

const LANCZOS_SUM_EXPG_SCALED_NUM: [f64; 13] = [
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859,
];

const LANCZOS_SUM_EXPG_SCALED_DENOM: [f64; 13] = [
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0,
];

fn lanczos_sum_expg_scaled(x: f64) -> f64 {
    ratevl(
        x,
        &LANCZOS_SUM_EXPG_SCALED_NUM,
        &LANCZOS_SUM_EXPG_SCALED_DENOM,
    )
}

/// Rational polynomial evaluation matching cephes `ratevl` (Horner with a
/// reciprocal flip for |x| > 1 to preserve precision).
fn ratevl(x: f64, num: &[f64], denom: &[f64]) -> f64 {
    let m = num.len() - 1;
    let n = denom.len() - 1;
    let absx = x.abs();

    let (y, hi_first) = if absx > 1.0 {
        (1.0 / x, true)
    } else {
        (x, false)
    };

    let coef =
        |arr: &[f64], top: usize, i: usize| -> f64 { if hi_first { arr[top - i] } else { arr[i] } };

    let mut num_ans = coef(num, m, 0);
    for i in 1..=m {
        num_ans = num_ans * y + coef(num, m, i);
    }
    let mut denom_ans = coef(denom, n, 0);
    for i in 1..=n {
        denom_ans = denom_ans * y + coef(denom, n, i);
    }

    if absx > 1.0 {
        x.powi((n as i32) - (m as i32)) * num_ans / denom_ans
    } else {
        num_ans / denom_ans
    }
}

// Log-gamma (gamma.c).

const LOGPI: f64 = 1.14472988584940017414;
const LS2PI: f64 = 0.91893853320467274178;

const LGAM_A: [f64; 5] = [
    8.11614167470508450300e-4,
    -5.95061904284301438324e-4,
    7.93650340457716943945e-4,
    -2.77777777730099687205e-3,
    8.33333333333331927722e-2,
];

const LGAM_B: [f64; 6] = [
    -1.37825152569120859100e3,
    -3.88016315134637840924e4,
    -3.31612992738871184744e5,
    -1.16237097492762307383e6,
    -1.72173700820839662146e6,
    -8.53555664245765465627e5,
];

const LGAM_C: [f64; 6] = [
    -3.51815701436523470549e2,
    -1.70642106651881159223e4,
    -2.20528590553854454839e5,
    -1.13933444367982507207e6,
    -2.53252307177582951285e6,
    -2.01889141433532773231e6,
];

const MAXLGM: f64 = 2.556348e305;

fn lgam(x: f64) -> f64 {
    lgam_sgn(x).0
}

fn lgam_sgn(x_in: f64) -> (f64, i32) {
    let mut sign = 1;
    if !x_in.is_finite() {
        return (x_in, sign);
    }
    let mut x = x_in;

    if x < -34.0 {
        let q = -x;
        let (w, _s) = lgam_sgn(q);
        let mut p = q.floor();
        if p == q {
            return (f64::INFINITY, sign);
        }
        let i = p as i64;
        sign = if (i & 1) == 0 { -1 } else { 1 };
        let mut z = q - p;
        if z > 0.5 {
            p += 1.0;
            z = p - q;
        }
        z = q * (std::f64::consts::PI * z).sin();
        if z == 0.0 {
            return (f64::INFINITY, sign);
        }
        z = LOGPI - z.ln() - w;
        return (z, sign);
    }

    if x < 13.0 {
        let mut z = 1.0;
        let mut p = 0.0;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return (f64::INFINITY, sign);
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if z < 0.0 {
            sign = -1;
            z = -z;
        } else {
            sign = 1;
        }
        if u == 2.0 {
            return (z.ln(), sign);
        }
        p -= 2.0;
        x += p;
        let pp = x * polevl(x, &LGAM_B) / p1evl(x, &LGAM_C);
        return (z.ln() + pp, sign);
    }

    if x > MAXLGM {
        return (sign as f64 * f64::INFINITY, sign);
    }

    let mut q = (x - 0.5) * x.ln() - x + LS2PI;
    if x > 1.0e8 {
        return (q, sign);
    }
    let p = 1.0 / (x * x);
    if x >= 1000.0 {
        q += ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p
            + 0.0833333333333333333333)
            / x;
    } else {
        q += polevl(p, &LGAM_A) / x;
    }
    (q, sign)
}

// Near-unity helpers (unity.c).

/// log(1 + x) - x, accurate near 0.
fn log1pmx(x: f64) -> f64 {
    if x.abs() < 0.5 {
        let mut xfac = x;
        let mut res = 0.0;
        for n in 2..MAXITER {
            xfac *= -x;
            let term = xfac / n as f64;
            res += term;
            if term.abs() < MACHEP * res.abs() {
                break;
            }
        }
        res
    } else {
        x.ln_1p() - x
    }
}

fn lgam1p_taylor(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut res = -EULER * x;
    let mut xfac = -x;
    for n in 2..42 {
        xfac *= -x;
        let coeff = ZETA[n - 2] * xfac / n as f64;
        res += coeff;
        if coeff.abs() < MACHEP * res.abs() {
            break;
        }
    }
    res
}

/// lgam(x + 1), accurate near 0 and 1.
fn lgam1p(x: f64) -> f64 {
    if x.abs() <= 0.5 {
        lgam1p_taylor(x)
    } else if (x - 1.0).abs() < 0.5 {
        x.ln() + lgam1p_taylor(x - 1.0)
    } else {
        lgam(x + 1.0)
    }
}

// Incomplete gamma (igam.c).

const IGAM_K: usize = 25;
const IGAM_N: usize = 25;
const BIG: f64 = 4.503599627370496e15;
const BIGINV: f64 = 2.22044604925031308085e-16;
const SMALL: f64 = 20.0;
const LARGE: f64 = 200.0;
const SMALLRATIO: f64 = 0.3;
const LARGERATIO: f64 = 4.5;

include!("igam_d.rs");

fn igam_fac(a: f64, x: f64) -> f64 {
    if (a - x).abs() > 0.4 * a.abs() {
        let ax = a * x.ln() - x - lgam(a);
        if ax < -MAXLOG {
            return 0.0;
        }
        return ax.exp();
    }

    let fac = a + LANCZOS_G - 0.5;
    let mut res = (fac / std::f64::consts::E).sqrt() / lanczos_sum_expg_scaled(a);

    if a < 200.0 && x < 200.0 {
        res *= (a - x).exp() * (x / fac).powf(a);
    } else {
        let num = x - a - LANCZOS_G + 0.5;
        res *= (a * log1pmx(num / fac) + x * (0.5 - LANCZOS_G) / fac).exp();
    }
    res
}

fn igamc_continued_fraction(a: f64, x: f64) -> f64 {
    let ax = igam_fac(a, x);
    if ax == 0.0 {
        return 0.0;
    }

    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0;
    let mut pkm2 = 1.0;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;

    for _ in 0..MAXITER {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;
        let t;
        if qk != 0.0 {
            let r = pk / qk;
            t = ((ans - r) / r).abs();
            ans = r;
        } else {
            t = 1.0;
        }
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if t <= MACHEP {
            break;
        }
    }
    ans * ax
}

fn igam_series(a: f64, x: f64) -> f64 {
    let ax = igam_fac(a, x);
    if ax == 0.0 {
        return 0.0;
    }
    let mut r = a;
    let mut c = 1.0;
    let mut ans = 1.0;
    for _ in 0..MAXITER {
        r += 1.0;
        c *= x / r;
        ans += c;
        if c <= MACHEP * ans {
            break;
        }
    }
    ans * ax / a
}

fn igamc_series(a: f64, x: f64) -> f64 {
    let mut fac = 1.0;
    let mut sum = 0.0;
    for n in 1..MAXITER {
        fac *= -x / n as f64;
        let term = fac / (a + n as f64);
        sum += term;
        if term.abs() <= MACHEP * sum.abs() {
            break;
        }
    }
    let logx = x.ln();
    let term = -(a * logx - lgam1p(a)).exp_m1();
    term - (a * logx - lgam(a)).exp() * sum
}

fn asymptotic_series(a: f64, x: f64, func_igam: bool) -> f64 {
    let lambda = x / a;
    let sigma = (x - a) / a;
    let sgn = if func_igam { -1.0 } else { 1.0 };

    let eta = if lambda > 1.0 {
        (-2.0 * log1pmx(sigma)).sqrt()
    } else if lambda < 1.0 {
        -(-2.0 * log1pmx(sigma)).sqrt()
    } else {
        0.0
    };
    let mut res = 0.5 * erfc(sgn * eta * (a / 2.0).sqrt());

    let mut etapow = [0.0f64; IGAM_N];
    etapow[0] = 1.0;
    let mut maxpow = 0usize;
    let mut sum = 0.0;
    let mut afac = 1.0;
    let mut absoldterm = f64::INFINITY;

    for k in 0..IGAM_K {
        let mut ck = IGAM_D[k][0];
        for n in 1..IGAM_N {
            if n > maxpow {
                etapow[n] = eta * etapow[n - 1];
                maxpow += 1;
            }
            let ckterm = IGAM_D[k][n] * etapow[n];
            ck += ckterm;
            if ckterm.abs() < MACHEP * ck.abs() {
                break;
            }
        }
        let term = ck * afac;
        let absterm = term.abs();
        if absterm > absoldterm {
            break;
        }
        sum += term;
        if absterm < MACHEP * sum.abs() {
            break;
        }
        absoldterm = absterm;
        afac /= a;
    }
    res += sgn * (-0.5 * a * eta * eta).exp() * sum / (2.0 * std::f64::consts::PI * a).sqrt();
    res
}

/// Regularized upper incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a)`. cephes `igamc`.
pub fn igamc(a: f64, x: f64) -> f64 {
    if x < 0.0 || a < 0.0 {
        return f64::NAN;
    } else if a == 0.0 {
        return if x > 0.0 { 0.0 } else { f64::NAN };
    } else if x == 0.0 {
        return 1.0;
    } else if a.is_infinite() {
        return if x.is_infinite() { f64::NAN } else { 1.0 };
    } else if x.is_infinite() {
        return 0.0;
    }

    let absxma_a = (x - a).abs() / a;
    if a > SMALL && a < LARGE && absxma_a < SMALLRATIO {
        return asymptotic_series(a, x, false);
    } else if a > LARGE && absxma_a < LARGERATIO / a.sqrt() {
        return asymptotic_series(a, x, false);
    }

    if x > 1.1 {
        if x < a {
            1.0 - igam_series(a, x)
        } else {
            igamc_continued_fraction(a, x)
        }
    } else if x <= 0.5 {
        if -0.4 / x.ln() < a {
            1.0 - igam_series(a, x)
        } else {
            igamc_series(a, x)
        }
    } else if x * 1.1 < a {
        1.0 - igam_series(a, x)
    } else {
        igamc_series(a, x)
    }
}

// erfc — needed by the incomplete-gamma asymptotic series (ndtr.c).

const ERF_P: [f64; 9] = [
    2.46196981473530512524e-10,
    5.64189564831068821977e-1,
    7.46321056442269912687e0,
    4.86371970985681366614e1,
    1.96520832956077098242e2,
    5.26445194995477358631e2,
    9.34528527171957607540e2,
    1.02755188689515710272e3,
    5.57535335369399327526e2,
];

const ERF_Q: [f64; 8] = [
    1.32281951154744992508e1,
    8.67072140885989742329e1,
    3.54937778887819891062e2,
    9.75708501743205489753e2,
    1.82390916687909736289e3,
    2.24633760818710981792e3,
    1.65666309194161350182e3,
    5.57535340817727675546e2,
];

const ERF_R: [f64; 6] = [
    5.64189583547755073984e-1,
    1.27536670759978104416e0,
    5.01905042251180477414e0,
    6.16021097993053585195e0,
    7.40974269950448939160e0,
    2.97886665372100240670e0,
];

const ERF_S: [f64; 6] = [
    2.26052863220117276590e0,
    9.39603524938001434673e0,
    1.20489539808096656605e1,
    1.70814450747565897222e1,
    9.60896809063285878198e0,
    3.36907645100081516050e0,
];

const ERF_T: [f64; 5] = [
    9.60497373987051638749e0,
    9.00260197203842689217e1,
    2.23200534594684319226e3,
    7.00332514112805075473e3,
    5.55923013010394962768e4,
];

const ERF_U: [f64; 5] = [
    3.35617141647503099647e1,
    5.21357949780152679795e2,
    4.59432382970980127987e3,
    2.26290000613890934246e4,
    4.92673942608635921086e4,
];

const ERF_MAXLOG: f64 = 7.09782712893383996843e2;

fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &ERF_T) / p1evl(z, &ERF_U)
}

fn erfc(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a.abs();
    if x < 1.0 {
        return 1.0 - erf(a);
    }
    let z = -a * a;
    if z < -ERF_MAXLOG {
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    let zexp = z.exp();
    let (p, q) = if x < 8.0 {
        (polevl(x, &ERF_P), p1evl(x, &ERF_Q))
    } else {
        (polevl(x, &ERF_R), p1evl(x, &ERF_S))
    };
    let mut y = (zexp * p) / q;
    if a < 0.0 {
        y = 2.0 - y;
    }
    if y == 0.0 {
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(got: f64, want: f64) -> f64 {
        (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn lgam_known() {
        assert!(rel(lgam(0.5), 0.5723649429247001) < 1e-14);
        assert!(rel(lgam(10.0), 12.801827480081469) < 1e-14);
    }

    #[test]
    fn igamc_chi2_sf_known() {
        // scipy.special.gammaincc / scipy.stats.chi2.sf(x, df) == igamc(df/2, x/2)
        assert!(rel(igamc(1.0, 1.5), 0.22313016014842982) < 1e-13);
        assert!(rel(igamc(2.5, 5.0), 0.07523524614651216) < 1e-13);
        assert!(rel(igamc(5.0, 4.618), 0.5098624956475126) < 1e-13);
    }
}
