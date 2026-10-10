//! PROBE: times opening a stage and traversing it, best of `runs`.
//!
//! ```bash
//! python3 crates/openusd/examples/par_probe_scene.py /tmp/large 50 100 10
//! cargo run --release -p openusd --features rayon --example par_probe -- /tmp/large/shot.usda 3
//! ```
//!
//! The environment picks what runs: `OPENUSD_PAR` is the fan-out strategy
//! (`rayon`, `std` or `off`), `RAYON_NUM_THREADS` the rayon pool size,
//! `PROBE_SUBTREE` composes each traversal's whole namespace one level at a
//! time before the walk, `PAR_STATS` prints the batch counters and `PROXIES`
//! walks instance proxies.
use std::env;
use std::time::Instant;

use openusd::usd::{PrimPredicate, Stage};

fn main() {
    let mut args = env::args().skip(1);
    let root = args.next().expect("root");
    let runs: usize = args.next().map_or(3, |n| n.parse().expect("runs"));
    let proxies = env::var("PROXIES").is_ok();
    let predicate = if proxies {
        PrimPredicate::DEFAULT_PROXIES
    } else {
        PrimPredicate::DEFAULT
    };
    let (mut open_best, mut walk_best, mut prims) = (f64::MAX, f64::MAX, 0);
    for _ in 0..runs {
        let start = Instant::now();
        let stage = Stage::open(&root).expect("open");
        let opened = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut n = 0;
        stage.traverse(predicate, |_| n += 1).expect("traverse");
        let walked = start.elapsed().as_secs_f64();
        open_best = open_best.min(opened);
        walk_best = walk_best.min(walked);
        prims = n;
        drop(stage);
    }
    println!("prims {prims:>6}  open {open_best:.3}s  traverse {walk_best:.3}s");
    if env::var("PAR_STATS").is_ok() {
        println!("  {}", openusd::pcp::par_stats());
    }
}
