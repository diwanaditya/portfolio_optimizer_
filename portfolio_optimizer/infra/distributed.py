"""
Distributed Execution.

SCOPE HONESTY: this uses Python's stdlib `multiprocessing` /
`concurrent.futures.ProcessPoolExecutor` to parallelize independent work
(parameter sweeps, multi-strategy backtests, multi-seed statistical
validation) across CPU cores on a SINGLE machine. It is NOT a distributed
computing framework — it does not span multiple machines, has no job
scheduler, no fault-tolerant task retry across nodes, and no shared
distributed state. For actual multi-machine distributed execution you'd
reach for Ray, Dask, or a cloud batch service; what's here is the correct
single-machine parallelism layer, which is often what's actually needed
(most parameter sweeps and backtest grids are embarrassingly parallel
and fit comfortably on one multi-core machine) and is a reasonable
foundation to swap in a real distributed backend behind, since the task
function signature here is identical to what Ray/Dask would expect.
"""
from __future__ import annotations
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class DistributedTaskResult:
    task_id: str
    result: Any
    runtime_seconds: float
    error: str | None = None


def run_parallel_tasks(task_fn: Callable, task_args: list, max_workers: int | None = None,
                        task_ids: list | None = None) -> list:
    """Runs `task_fn(*args)` for each entry in `task_args` across a
    process pool, returning results in the SAME order as the input
    (not completion order), so callers don't need to track task
    identity manually for simple sweeps.

    Parameters
    ----------
    task_fn : must be a module-level (picklable) function -- this is a
              real constraint of multiprocessing, not an oversight; a
              lambda or closure will fail to pickle across processes.
    task_args : list of argument tuples, one per task
    max_workers : defaults to os.cpu_count()
    task_ids : optional labels for each task, for readable result tracking
    """
    task_ids = task_ids or [str(i) for i in range(len(task_args))]
    results = [None] * len(task_args)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, args in enumerate(task_args):
            t0 = time.perf_counter()
            future = executor.submit(_run_and_time, task_fn, args, t0)
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result, runtime = future.result()
                results[idx] = DistributedTaskResult(
                    task_id=task_ids[idx], result=result, runtime_seconds=runtime,
                )
            except Exception as e:
                results[idx] = DistributedTaskResult(
                    task_id=task_ids[idx], result=None, runtime_seconds=0.0, error=str(e),
                )
    return results


def _run_and_time(task_fn, args, start_time):
    result = task_fn(*args)
    return result, time.perf_counter() - start_time


class ParallelParameterSweep:
    """Convenience wrapper for the common case: sweep one parameter
    across many values, running each in a separate process. Useful when
    a single sweep point is itself expensive (e.g. a Michaud-resampled
    frontier with hundreds of bootstrap draws, or a multi-seed regime
    robustness test), where single-process sequential sweeping would be
    the actual bottleneck.
    """

    def __init__(self, task_fn: Callable, max_workers: int | None = None):
        self.task_fn = task_fn
        self.max_workers = max_workers

    def run(self, parameter_values: list) -> dict:
        task_args = [(v,) for v in parameter_values]
        task_ids = [str(v) for v in parameter_values]
        results = run_parallel_tasks(self.task_fn, task_args, self.max_workers, task_ids)
        return {r.task_id: r for r in results}
