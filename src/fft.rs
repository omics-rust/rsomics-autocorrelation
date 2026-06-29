//! FFT autocovariance — the `fft=True` path of `statsmodels acovf`.
//!
//! statsmodels computes the autocovariances by circular convolution:
//! `ifft(|fft(xo)|²)`, zero-padding to a length ≥ `2·nobs+1`. We pad to the
//! next power of two (linear autocorrelation is pad-length-independent for any
//! pad ≥ `2·nobs-1`) and use our own radix-2 FFT. The result matches
//! statsmodels `fft=True` only to ~1e-10 (FFT arithmetic ordering differs from
//! numpy/pocketfft, and is not bit-portable); the direct path is the
//! value-exact target.

use crate::sum::pairwise_sum;

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    fn zero() -> Self {
        C { re: 0.0, im: 0.0 }
    }
    fn mul(self, o: C) -> C {
        C {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    fn add(self, o: C) -> C {
        C {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    fn sub(self, o: C) -> C {
        C {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

/// FFT-based autocovariance at lags `0..=nlags`, demeaned, divided by `n`
/// (biased) or `n-k` (adjusted).
pub fn acovf_fft(x: &[f64], nlags: usize, adjusted: bool) -> Vec<f64> {
    let n = x.len();
    let mean = pairwise_sum(x) / n as f64;

    let mut size = 1usize;
    while size < 2 * n + 1 {
        size <<= 1;
    }

    let mut buf = vec![C::zero(); size];
    for (b, &v) in buf.iter_mut().zip(x) {
        b.re = v - mean;
    }

    fft(&mut buf, false);
    for c in buf.iter_mut() {
        let p = c.re * c.re + c.im * c.im;
        c.re = p;
        c.im = 0.0;
    }
    fft(&mut buf, true);

    let mut acov = vec![0.0; nlags + 1];
    for (k, a) in acov.iter_mut().enumerate() {
        let v = buf[k].re / size as f64;
        *a = v / if adjusted { (n - k) as f64 } else { n as f64 };
    }
    acov
}

fn fft(a: &mut [C], inverse: bool) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let ang = 2.0 * std::f64::consts::PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let wlen = C {
            re: ang.cos(),
            im: ang.sin(),
        };
        let mut i = 0;
        while i < n {
            let mut w = C { re: 1.0, im: 0.0 };
            for k in 0..len / 2 {
                let u = a[i + k];
                let v = a[i + k + len / 2].mul(w);
                a[i + k] = u.add(v);
                a[i + k + len / 2] = u.sub(v);
                w = w.mul(wlen);
            }
            i += len;
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::acovf_fft;
    use crate::acf::acovf;

    #[test]
    fn fft_matches_direct_to_tolerance() {
        let x: Vec<f64> = (0..500)
            .map(|i| (i as f64 * 0.17).sin() + 0.01 * i as f64)
            .collect();
        let d = acovf(&x, 20, false);
        let f = acovf_fft(&x, 20, false);
        for (a, b) in d.iter().zip(&f) {
            assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
}
