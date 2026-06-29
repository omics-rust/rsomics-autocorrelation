//! Autocorrelation (ACF), partial autocorrelation (PACF), and Ljung-Box /
//! Box-Pierce Q-tests — value-exact to statsmodels 0.14.6 (`tsa.stattools` and
//! `stats.diagnostic`), to a few ULP.
//!
//! The autocovariances follow the direct (`fft=False`) estimator with numpy's
//! pairwise summation. statsmodels routes the lag dot-products through BLAS, so
//! the agreement is a few ULP (~1e-15 relative) rather than bit-for-bit; the
//! Durbin-Levinson PACF recursion is bit-faithful given the same autocovariances.

pub mod acf;
pub mod fft;
pub mod igamc;
pub mod io;
pub mod ljungbox;
pub mod ndtr;
pub mod pacf;
mod sum;

pub use acf::{acf, acovf, bartlett_halfwidth, default_acf_nlags, q_stat};
pub use fft::acovf_fft;
pub use io::read_values;
pub use ljungbox::{LjungBox, acorr_ljungbox, default_ljungbox_lags};
pub use pacf::{Method, default_pacf_nlags, pacf, pacf_halfwidth};
