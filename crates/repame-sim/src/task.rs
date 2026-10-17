//! A fixed-size job pool for off-thread work.
//!
//! Used for the jobs that are naturally owned and independent: asset import,
//! chunk streaming, navmesh bake, atlas packing. Jobs are `'static` (they own
//! their inputs and return their output); for work that must borrow a local
//! slice, use `std::thread::scope` directly — this pool is not a scoped pool.
//!
//! Quality-of-life matched to the reference task pools (bevy_tasks, Godot's
//! `WorkerThreadPool`): a worker that panics carries the panic to the joiner
//! rather than poisoning the queue or silently dropping the job, and the pool
//! joins its workers on drop.
//!
//! Determinism: a job pool introduces scheduling nondeterminism by nature.
//! Never let a sim outcome depend on the order in which jobs finish; collect
//! results and apply them in a fixed order (or in a `Sim` schedule that reads
//! a completed queue) if the result feeds gameplay.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use bevy_ecs::prelude::Resource;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Queue {
    jobs: VecDeque<Job>,
    closed: bool,
}

struct Inner {
    queue: Mutex<Queue>,
    signal: Condvar,
}

enum SlotState<T> {
    Empty,
    Done(T),
    Failed(Box<dyn Any + Send + 'static>),
}

struct Slot<T> {
    value: Mutex<SlotState<T>>,
    ready: Condvar,
}

/// A handle to a job running on a [`TaskPool`].
///
/// Dropping the handle does not cancel the job. `join` waits for it and
/// re-raises a panic the job raised.
pub struct Task<T> {
    slot: Arc<Slot<T>>,
}

impl<T> Task<T> {
    /// Wait for the job and return its value, re-raising any panic.
    pub fn join(self) -> T {
        let mut guard = self.slot.value.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if !matches!(&*guard, SlotState::Empty) {
                break;
            }
            guard = self
                .slot
                .ready
                .wait(guard)
                .unwrap_or_else(|e| e.into_inner());
        }
        match std::mem::replace(&mut *guard, SlotState::Empty) {
            SlotState::Done(value) => value,
            SlotState::Failed(payload) => std::panic::resume_unwind(payload),
            SlotState::Empty => unreachable!("wait returned on an empty slot"),
        }
    }

    /// Whether the job has finished. A finished job may still be panicked.
    pub fn is_finished(&self) -> bool {
        !matches!(
            &*self.slot.value.lock().unwrap_or_else(|e| e.into_inner()),
            SlotState::Empty
        )
    }
}

/// Fixed-size worker pool. `new(n)` starts `n` worker threads.
///
/// Dropping the pool stops the workers once they drain the queue, joins them,
/// and abandons any job still queued (its [`Task`] never resolves). Keep the
/// pool alive until the tasks you joined on have finished.
#[derive(Resource)]
pub struct TaskPool {
    inner: Arc<Inner>,
    workers: Vec<JoinHandle<()>>,
}

impl TaskPool {
    /// Start a pool with `threads` workers. At least one worker is started.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let inner = Arc::new(Inner {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                closed: false,
            }),
            signal: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(threads);
        for index in 0..threads {
            let inner = Arc::clone(&inner);
            let handle = std::thread::Builder::new()
                .name(format!("repame-task-{index}"))
                .spawn(move || worker(inner))
                .expect("failed to start a task-pool worker");
            workers.push(handle);
        }
        Self { inner, workers }
    }

    /// A pool sized to the machine's available parallelism, at least one.
    pub fn auto() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self::new(threads)
    }

    /// Number of workers.
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// Queue a job and return a handle to its result.
    pub fn spawn<F, T>(&self, job: F) -> Task<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let slot = Arc::new(Slot {
            value: Mutex::new(SlotState::Empty),
            ready: Condvar::new(),
        });
        let done = Arc::clone(&slot);
        self.push(Box::new(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
            let mut guard = done.value.lock().unwrap_or_else(|e| e.into_inner());
            *guard = match outcome {
                Ok(value) => SlotState::Done(value),
                Err(payload) => SlotState::Failed(payload),
            };
            done.ready.notify_all();
        }));
        Task { slot }
    }

    /// Queue every job and return their results in input order.
    pub fn map<I, F, T>(&self, items: I, job: F) -> Vec<T>
    where
        I: IntoIterator,
        F: Fn(I::Item) -> T + Send + Sync + 'static,
        I::Item: Send + 'static,
        T: Send + 'static,
    {
        let job = Arc::new(job);
        let mut tasks = Vec::new();
        for item in items {
            let job = Arc::clone(&job);
            tasks.push(self.spawn(move || job(item)));
        }
        tasks.into_iter().map(Task::join).collect()
    }

    fn push(&self, job: Job) {
        let mut queue = self.inner.queue.lock().unwrap_or_else(|e| e.into_inner());
        if queue.closed {
            return;
        }
        queue.jobs.push_back(job);
        self.inner.signal.notify_one();
    }
}

impl Drop for TaskPool {
    fn drop(&mut self) {
        {
            let mut queue = self.inner.queue.lock().unwrap_or_else(|e| e.into_inner());
            queue.closed = true;
        }
        self.inner.signal.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker(inner: Arc<Inner>) {
    loop {
        let job = {
            let mut queue = inner.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(job) = queue.jobs.pop_front() {
                    break Some(job);
                }
                if queue.closed {
                    break None;
                }
                queue = inner.signal.wait(queue).unwrap_or_else(|e| e.into_inner());
            }
        };
        match job {
            Some(job) => job(),
            None => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn spawn_joins_to_the_value() {
        let pool = TaskPool::new(2);
        assert_eq!(pool.spawn(|| 6 * 7).join(), 42);
        assert_eq!(pool.threads(), 2);
    }

    #[test]
    fn one_worker_is_the_floor() {
        let pool = TaskPool::new(0);
        assert_eq!(pool.threads(), 1);
        assert_eq!(pool.spawn(|| 1 + 1).join(), 2);
    }

    #[test]
    fn map_preserves_input_order() {
        let pool = TaskPool::new(4);
        let out = pool.map(0..64, |n| n * n);
        assert_eq!(out, (0..64).map(|n| n * n).collect::<Vec<_>>());
    }

    #[test]
    fn jobs_actually_run_off_thread() {
        let pool = TaskPool::new(3);
        let counter = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..32 {
            let counter = Arc::clone(&counter);
            tasks.push(pool.spawn(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }));
        }
        for task in tasks {
            task.join();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 32);
    }

    #[test]
    #[should_panic(expected = "boom")]
    fn a_panicking_job_re_raises_at_the_joiner() {
        let pool = TaskPool::new(1);
        pool.spawn(|| panic!("boom")).join();
    }

    #[test]
    fn a_panicked_worker_does_not_stop_the_pool() {
        let pool = TaskPool::new(2);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.spawn(|| panic!("boom")).join();
        }));
        assert!(failed.is_err());
        assert_eq!(pool.spawn(|| 3).join(), 3);
    }

    #[test]
    fn dropping_the_pool_joins_its_workers() {
        let pool = TaskPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let task = pool.spawn(move || {
            c.fetch_add(1, Ordering::SeqCst);
        });
        task.join();
        drop(pool);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
