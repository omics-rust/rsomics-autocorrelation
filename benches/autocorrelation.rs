use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_autocorrelation::{acf::acf, acorr_ljungbox, pacf::Method, pacf::pacf};
use std::hint::black_box;

/// Deterministic AR(1) series via an LCG-driven innovation, matching the compat
/// generator so the autocorrelation structure is non-trivial.
fn ar1(n: usize) -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut prev = 0.0;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            let e = u - 0.5;
            prev = 0.7 * prev + e;
            prev
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let data = ar1(1_000_000);
    c.bench_function("acf_1e6_nlags40", |b| {
        b.iter(|| acf(black_box(&data), 40, false))
    });
    c.bench_function("pacf_1e6_nlags40_yw", |b| {
        b.iter(|| pacf(black_box(&data), 40, Method::Yw))
    });
    c.bench_function("ljungbox_1e6_lags40", |b| {
        b.iter(|| acorr_ljungbox(black_box(&data), 40, false, 0))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
