//! Fan-out of independent per-prim work across threads.
//!
//! PROBE: the strategy is picked at run time from `OPENUSD_PAR` (`off`, `std`,
//! or `rayon`, the default when the `rayon` feature is on) so one binary can
//! measure all three.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::{env, thread};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Off,
    Std,
    #[cfg(feature = "rayon")]
    Rayon,
}

fn strategy() -> Strategy {
    static STRATEGY: OnceLock<Strategy> = OnceLock::new();
    *STRATEGY.get_or_init(|| match env::var("OPENUSD_PAR").as_deref() {
        Ok("off") => Strategy::Off,
        Ok("std") => Strategy::Std,
        #[cfg(feature = "rayon")]
        _ => Strategy::Rayon,
        #[cfg(not(feature = "rayon"))]
        _ => Strategy::Std,
    })
}

/// Applies `f` to every item, in parallel when there is more than one, and
/// returns the results in item order.
pub(super) fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    let wall = Instant::now();
    let f = |item: &T| {
        let start = Instant::now();
        let r = f(item);
        WORK_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    };
    let out = map_inner(items, f);
    BATCHES.fetch_add(1, Ordering::Relaxed);
    ITEMS.fetch_add(items.len() as u64, Ordering::Relaxed);
    if items.len() >= 2 {
        PAR_BATCHES.fetch_add(1, Ordering::Relaxed);
        PAR_ITEMS.fetch_add(items.len() as u64, Ordering::Relaxed);
    }
    WALL_NS.fetch_add(wall.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

static BATCHES: AtomicU64 = AtomicU64::new(0);
static ITEMS: AtomicU64 = AtomicU64::new(0);
static PAR_BATCHES: AtomicU64 = AtomicU64::new(0);
static PAR_ITEMS: AtomicU64 = AtomicU64::new(0);
static WALL_NS: AtomicU64 = AtomicU64::new(0);
static WORK_NS: AtomicU64 = AtomicU64::new(0);

/// PROBE counters.
#[doc(hidden)]
pub fn stats() -> String {
    let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
    format!(
        "batches {} items {} | fanned batches {} items {} | wall in map {:.3}s, work in closures {:.3}s",
        get(&BATCHES),
        get(&ITEMS),
        get(&PAR_BATCHES),
        get(&PAR_ITEMS),
        get(&WALL_NS) as f64 / 1e9,
        get(&WORK_NS) as f64 / 1e9
    )
}

fn map_inner<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    if items.len() < 2 {
        return items.iter().map(f).collect();
    }
    match strategy() {
        Strategy::Off => items.iter().map(f).collect(),
        Strategy::Std => {
            let threads = thread::available_parallelism().map_or(1, |n| n.get()).min(items.len());
            let chunk = items.len().div_ceil(threads);
            thread::scope(|scope| {
                let handles: Vec<_> = items
                    .chunks(chunk)
                    .map(|chunk| scope.spawn(|| chunk.iter().map(&f).collect::<Vec<_>>()))
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|handle| handle.join().expect("a composition worker panicked"))
                    .collect()
            })
        }
        #[cfg(feature = "rayon")]
        Strategy::Rayon => {
            use rayon::prelude::*;
            items.par_iter().map(f).collect()
        }
    }
}
