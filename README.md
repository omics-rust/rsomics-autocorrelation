# rsomics-autocorrelation

Autocorrelation (ACF), partial autocorrelation (PACF), and Ljung-Box /
Box-Pierce Q-tests of a time series — a value-exact Rust port of
`statsmodels.tsa.stattools` (`acf`, `pacf`, `acovf`) and
`statsmodels.stats.diagnostic.acorr_ljungbox` (statsmodels 0.14.6).

```text
rsomics-autocorrelation <data.tsv> --stat acf|pacf|ljungbox [options] [--json]
```

The input is a whitespace-separated column of numbers (`-` reads stdin).
`--stat` selects the quantity; defaults follow statsmodels exactly.

## Operations

`--stat acf` — autocorrelation at lags `0..=nlags`. Output: `lag<TAB>acf`, with
`<TAB>lower<TAB>upper` appended under `--alpha A` (Bartlett-formula confidence
intervals centred on the ACF) and `<TAB>qstat<TAB>qpvalue` appended under
`--qstat` (Ljung-Box Q and its chi-square p-value, or Box-Pierce Q under
`--boxpierce`). `--adjusted` divides the autocovariance by `n-k` instead of `n`.

`--stat pacf` — partial autocorrelation at lags `0..=nlags` via `--method`:

| `--method` | statsmodels method | estimator |
|---|---|---|
| `yw` (default) | `ywadjusted` | Yule-Walker, `n-k` autocovariance denominator |
| `ywm` | `ywmle` | Yule-Walker, `n` denominator |
| `ld` | `ldadjusted` | Levinson-Durbin, `n-k` denominator |
| `ldb` | `ldbiased` | Levinson-Durbin, `n` denominator |
| `ols` | `ols` (efficient) | OLS of the series on its lags plus a constant |

`--alpha A` adds `1/sqrt(n)` confidence bounds.

`--stat ljungbox` — Ljung-Box `Q_k = n(n+2)·Σ_{j≤k} acf[j]²/(n-j)` for lags
`1..=nlags`, with `p = chi2.sf(Q_k, k)`. `--boxpierce` additionally reports the
Box-Pierce statistic `Q_k = n·Σ acf[j]²`. Output:
`lag<TAB>lb_stat<TAB>lb_pvalue` (`<TAB>bp_stat<TAB>bp_pvalue` with `--boxpierce`).

### Defaults matched to statsmodels

- `acf` `nlags`: `min(int(10·log10(n)), n-1)`.
- `pacf` `nlags`: `max(min(int(10·log10(n)), n//2-1), 1)`, default `method="yw"`
  (statsmodels `"ywadjusted"`).
- `ljungbox` `lags`: `min(10, n//5)`.
- ACF/PACF autocovariance is the direct (`fft=False`), demeaned, biased
  (`adjusted=False`) estimator.

`--fft` computes the ACF via an internal radix-2 FFT. It matches statsmodels
`fft=True` only to ~1e-10 (FFT arithmetic ordering, not bit-portable); the
value-exact target is the default direct path.

## Value-exactness

statsmodels computes the direct autocovariance as `acov[k] = xo[k:]·xo[:-k]`,
demeaning with `x - x.mean()`. The mean uses numpy's pairwise summation, which
this crate reproduces bit-for-bit; the lag dot-products, however, are dispatched
by numpy to **BLAS `ddot`**, whose SIMD reduction order is internal to the BLAS
build and is *not* portable across implementations or CPU architectures.
Consequently the autocovariances — and everything derived from them (ACF, PACF,
the Ljung-Box / Box-Pierce Q) — agree with statsmodels to **a few ULP (~1e-15
relative), not 0 ULP**. This crate uses numpy's pairwise summation for the lag
products, the closest portable order; the residual is dominated by BLAS rounding.

Given the same autocovariances, the PACF Levinson-Durbin recursion (`ld`/`ldb`)
is bit-faithful to statsmodels' own recursion, and the Yule-Walker methods
(`yw`/`ywm`) reach the same Yule-Walker solution that statsmodels obtains via a
LAPACK `solve`, agreeing to ~1e-14. The chi-square p-values and the Bartlett /
1-√n confidence bounds go through ported cephes transcendental functions
(`igamc`, `ndtri`) and match scipy to ~1e-13.

`tests/compat.rs` re-derives committed statsmodels goldens — AR(1) series at
n=50 / n=5000 / n=10⁶ and a white-noise n=2000 series, across all ACF/PACF/Q
options — with no statsmodels present, asserting every numeric column to 1e-12
relative. Goldens are stored as IEEE-754 hex bit patterns, because serde_json's
float parser is not correctly-rounded and would otherwise mask the comparison.

## Origin

This crate is an independent Rust reimplementation based on:

- The statsmodels 0.14.6 source (`statsmodels/tsa/stattools.py` — `acovf`,
  `acf`, `pacf`, `pacf_yw`, `pacf_ols`, `levinson_durbin`, `q_stat`; and
  `statsmodels/stats/diagnostic.py` — `acorr_ljungbox`; with
  `statsmodels/regression/linear_model.py` — `yule_walker`), BSD-3-Clause,
  which is permissively licensed and was read and cited for the exact formulas,
  denominators, default-lag rules, and column semantics.
- The numpy reduction semantics for `np.mean` (`pairwise_sum`), reproduced so
  the demeaning matches at large N.
- Stephen L. Moshier's Cephes Math Library (public domain), as vendored in SciPy
  (`scipy/special/cephes/` — `igam.c`, `gamma.c`, `unity.c`, `ndtr.c`), BSD-3,
  for the regularized upper incomplete gamma `igamc` (chi-square survival
  function) and the inverse normal CDF `ndtri` (confidence-interval quantile).
  Algorithms and coefficients are transcribed verbatim; only the control flow is
  rewritten in Rust.

License: MIT OR Apache-2.0.
Upstream credit: statsmodels https://github.com/statsmodels/statsmodels
(BSD-3-Clause); cephes / SciPy https://github.com/scipy/scipy (BSD-3-Clause).
