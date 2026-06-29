use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};
use serde::Serialize;

use rsomics_autocorrelation::{
    acf::{acovf, bartlett_halfwidth, default_acf_nlags, q_stat},
    acovf_fft,
    ljungbox::{acorr_ljungbox, default_ljungbox_lags},
    pacf::{Method, default_pacf_nlags, pacf, pacf_halfwidth},
    read_values,
};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Stat {
    Acf,
    Pacf,
    Ljungbox,
}

/// Autocorrelation (ACF), partial autocorrelation (PACF), and Ljung-Box /
/// Box-Pierce Q-tests of a time series (`statsmodels.tsa.stattools` /
/// `statsmodels.stats.diagnostic`).
///
/// Reads a whitespace-separated column of numbers (`-` reads stdin). `--stat`
/// selects the quantity. ACF/PACF print `lag<TAB>value` (with confidence
/// bounds and Q columns appended when requested); Ljung-Box prints
/// `lag<TAB>lb_stat<TAB>lb_pvalue` (plus Box-Pierce columns with
/// `--boxpierce`).
#[derive(Parser, Debug)]
#[command(name = "rsomics-autocorrelation", version, about, long_about = None)]
pub struct Cli {
    /// Input value TSV; one observation per line (`-` reads stdin).
    #[arg(value_name = "DATA", default_value = "-")]
    pub data: PathBuf,

    /// Quantity to compute.
    #[arg(long, value_enum, default_value_t = Stat::Acf)]
    pub stat: Stat,

    /// Number of lags (statsmodels default rule per stat when omitted).
    #[arg(long)]
    pub nlags: Option<usize>,

    /// Use the n-k autocovariance denominator (statsmodels `adjusted`).
    #[arg(long)]
    pub adjusted: bool,

    /// PACF method (statsmodels `pacf` default `yw` = `ywadjusted`).
    #[arg(long, default_value = "yw")]
    pub method: String,

    /// Append Ljung-Box Q statistic and p-value columns to the ACF output.
    #[arg(long)]
    pub qstat: bool,

    /// Confidence level: emit `1-alpha` interval bounds (e.g. `0.05` → 95%).
    #[arg(long)]
    pub alpha: Option<f64>,

    /// Report the Box-Pierce statistic instead of / alongside Ljung-Box.
    #[arg(long)]
    pub boxpierce: bool,

    /// ACF via FFT (own FFT; matches statsmodels `fft=True` only to ~1e-10).
    #[arg(long)]
    pub fft: bool,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Serialize)]
struct AcfRow {
    lag: usize,
    acf: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    lower: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upper: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    qstat: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    qpvalue: Option<f64>,
}

#[derive(Serialize)]
struct PacfRow {
    lag: usize,
    pacf: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    lower: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upper: Option<f64>,
}

#[derive(Serialize)]
struct LjungRow {
    lag: usize,
    lb_stat: f64,
    lb_pvalue: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    bp_stat: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bp_pvalue: Option<f64>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Output {
    Acf { acf: Vec<AcfRow> },
    Pacf { pacf: Vec<PacfRow> },
    Ljungbox { ljungbox: Vec<LjungRow> },
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let x = read_values(&self.data)?;
            let n = x.len();
            match self.stat {
                Stat::Acf => self.run_acf(&x, n, common.json),
                Stat::Pacf => self.run_pacf(&x, n, common.json),
                Stat::Ljungbox => self.run_ljungbox(&x, n, common.json),
            }
        })
    }

    fn run_acf(&self, x: &[f64], n: usize, json: bool) -> rsomics_common::Result<Output> {
        let nlags = self.nlags.unwrap_or_else(|| default_acf_nlags(n));
        if nlags > n - 1 {
            return Err(RsomicsError::InvalidInput(format!(
                "nlags {nlags} must be <= nobs-1 ({})",
                n - 1
            )));
        }
        let avf = if self.fft {
            acovf_fft(x, nlags, self.adjusted)
        } else {
            acovf(x, nlags, self.adjusted)
        };
        let a0 = avf[0];
        let acf: Vec<f64> = avf.iter().map(|a| a / a0).collect();

        let half = self.alpha.map(|a| bartlett_halfwidth(&acf, n, a));
        let (qs, qp) = if self.qstat {
            let (q, p) = q_stat(&acf[1..], n, self.boxpierce, 0);
            (Some(q), Some(p))
        } else {
            (None, None)
        };

        let mut rows = Vec::with_capacity(nlags + 1);
        for (k, &rk) in acf.iter().enumerate() {
            rows.push(AcfRow {
                lag: k,
                acf: rk,
                lower: half.as_ref().map(|h| rk - h[k]),
                upper: half.as_ref().map(|h| rk + h[k]),
                qstat: if k >= 1 {
                    qs.as_ref().map(|q| q[k - 1])
                } else {
                    None
                },
                qpvalue: if k >= 1 {
                    qp.as_ref().map(|p| p[k - 1])
                } else {
                    None
                },
            });
        }

        if !json {
            for r in &rows {
                print!("{}\t{}", r.lag, r.acf);
                if let (Some(lo), Some(hi)) = (r.lower, r.upper) {
                    print!("\t{lo}\t{hi}");
                }
                if let (Some(q), Some(p)) = (r.qstat, r.qpvalue) {
                    print!("\t{q}\t{p}");
                }
                println!();
            }
        }
        Ok(Output::Acf { acf: rows })
    }

    fn run_pacf(&self, x: &[f64], n: usize, json: bool) -> rsomics_common::Result<Output> {
        let method = Method::parse(&self.method).ok_or_else(|| {
            RsomicsError::InvalidInput(format!("unknown pacf method '{}'", self.method))
        })?;
        let nlags = self.nlags.unwrap_or_else(|| default_pacf_nlags(n)).max(1);
        if nlags > n / 2 {
            return Err(RsomicsError::InvalidInput(format!(
                "nlags {nlags} must be <= nobs/2 ({})",
                n / 2
            )));
        }
        let p = pacf(x, nlags, method);
        let half = self.alpha.map(|a| pacf_halfwidth(n, a));

        let mut rows = Vec::with_capacity(nlags + 1);
        for (k, &pk) in p.iter().enumerate() {
            let (lo, hi) = match half {
                Some(h) if k >= 1 => (Some(pk - h), Some(pk + h)),
                Some(_) => (Some(pk), Some(pk)),
                None => (None, None),
            };
            rows.push(PacfRow {
                lag: k,
                pacf: pk,
                lower: lo,
                upper: hi,
            });
        }

        if !json {
            for r in &rows {
                print!("{}\t{}", r.lag, r.pacf);
                if let (Some(lo), Some(hi)) = (r.lower, r.upper) {
                    print!("\t{lo}\t{hi}");
                }
                println!();
            }
        }
        Ok(Output::Pacf { pacf: rows })
    }

    fn run_ljungbox(&self, x: &[f64], n: usize, json: bool) -> rsomics_common::Result<Output> {
        let maxlag = self.nlags.unwrap_or_else(|| default_ljungbox_lags(n));
        if maxlag == 0 || maxlag > n - 1 {
            return Err(RsomicsError::InvalidInput(format!(
                "lags {maxlag} must be in 1..=nobs-1 ({})",
                n - 1
            )));
        }
        let r = acorr_ljungbox(x, maxlag, self.boxpierce, 0);

        let mut rows = Vec::with_capacity(maxlag);
        for i in 0..maxlag {
            rows.push(LjungRow {
                lag: r.lags[i],
                lb_stat: r.lb_stat[i],
                lb_pvalue: r.lb_pvalue[i],
                bp_stat: r.bp_stat.as_ref().map(|q| q[i]),
                bp_pvalue: r.bp_pvalue.as_ref().map(|p| p[i]),
            });
        }

        if !json {
            for r in &rows {
                print!("{}\t{}\t{}", r.lag, r.lb_stat, r.lb_pvalue);
                if let (Some(q), Some(p)) = (r.bp_stat, r.bp_pvalue) {
                    print!("\t{q}\t{p}");
                }
                println!();
            }
        }
        Ok(Output::Ljungbox { ljungbox: rows })
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
