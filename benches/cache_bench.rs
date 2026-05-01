use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use ferrocache::config::HnswConfig;
use ferrocache::index::SemanticIndex;
use ferrocache::wal::{Wal, WalEntry};

const DIM: usize = 384;

/// Deterministic LCG-based pseudo-random unit vectors.
fn vec_from_seed(seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut v = Vec::with_capacity(DIM);
    let mut sum_sq = 0.0f32;
    for _ in 0..DIM {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = ((state >> 33) as u32) as f32 / u32::MAX as f32 - 0.5;
        v.push(f);
        sum_sq += f * f;
    }
    let norm = sum_sq.sqrt().max(1e-9);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn make_entry(seed: u64) -> WalEntry {
    WalEntry {
        uuid: format!("u-{seed}"),
        embedding: vec_from_seed(seed),
        response: format!("response-{seed}"),
        query_text: format!("query-{seed}"),
    }
}

fn populate(n: usize) -> SemanticIndex {
    let mut idx = SemanticIndex::new(&HnswConfig::default());
    for i in 0..n {
        idx.replay_entry(make_entry(i as u64)).unwrap();
    }
    idx
}

fn bench_insert(c: &mut Criterion) {
    c.bench_function("insert_384d", |b| {
        let mut counter: u64 = 0;
        b.iter_batched(
            || {
                counter = counter.wrapping_add(1);
                (
                    SemanticIndex::new(&HnswConfig::default()),
                    make_entry(counter),
                )
            },
            |(mut idx, entry)| {
                idx.replay_entry(black_box(entry)).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_query_hit(c: &mut Criterion) {
    let idx = populate(1000);
    let probe = vec_from_seed(42);
    c.bench_function("query_hit_1k_384d", |b| {
        b.iter(|| {
            let hit = idx.query(black_box(&probe), 0.90).unwrap();
            black_box(hit);
        });
    });
}

fn bench_query_miss(c: &mut Criterion) {
    let idx = populate(1000);
    // Use a seed outside the populated range so it's not an exact match.
    let probe = vec_from_seed(999_999);
    c.bench_function("query_miss_1k_384d", |b| {
        b.iter(|| {
            let res = idx.query(black_box(&probe), 0.999).unwrap();
            black_box(res);
        });
    });
}

fn bench_insert_with_wal(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    c.bench_function("insert_wal_fsync_384d", |b| {
        let mut counter: u64 = 0;
        b.iter_batched(
            || {
                counter = counter.wrapping_add(1);
                let dir = tempfile::tempdir().unwrap();
                let wal_path = dir.path().join("bench.wal");
                let wal = runtime.block_on(Wal::open(&wal_path)).unwrap();
                let idx = SemanticIndex::new(&HnswConfig::default());
                (dir, wal, idx, make_entry(counter))
            },
            |(_dir, mut wal, mut idx, entry)| {
                runtime.block_on(async {
                    wal.append(&entry).await.unwrap();
                });
                idx.replay_entry(entry).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_insert,
    bench_query_hit,
    bench_query_miss,
    bench_insert_with_wal
);
criterion_main!(benches);
