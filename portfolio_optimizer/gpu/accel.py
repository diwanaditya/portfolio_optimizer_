"""
GPU Acceleration Layer.

Where GPU acceleration actually pays off in this codebase is the
*embarrassingly parallel, matrix-heavy* workloads: Monte Carlo tail-risk
simulation (`advanced/stress_testing.py`), Michaud resampling
(`advanced/robust_frontier.py`), and scenario generation for multi-period
MPC. Optimization itself (SLSQP/linprog) is inherently sequential and
doesn't benefit from a GPU — so this module deliberately targets only the
simulation-heavy pieces rather than pretending to GPU-accelerate the whole
library.

Backend priority: CuPy > JAX > PyTorch > NumPy (CPU fallback). Every
function here degrades gracefully to plain NumPy if no GPU library is
installed or no GPU is present — nothing in the rest of the package
requires a GPU to run.
"""
from __future__ import annotations
import numpy as np


def available_backends() -> dict:
    backends = {"numpy": True}
    try:
        import cupy  # noqa: F401
        backends["cupy"] = True
    except ImportError:
        backends["cupy"] = False
    try:
        import jax  # noqa: F401
        backends["jax"] = True
    except ImportError:
        backends["jax"] = False
    try:
        import torch
        backends["torch"] = True
        backends["torch_cuda"] = torch.cuda.is_available()
    except ImportError:
        backends["torch"] = False
        backends["torch_cuda"] = False
    return backends


def best_backend(prefer: str | None = None) -> str:
    avail = available_backends()
    if prefer and avail.get(prefer):
        return prefer
    if avail.get("cupy"):
        return "cupy"
    if avail.get("jax"):
        return "jax"
    if avail.get("torch_cuda"):
        return "torch"
    return "numpy"


class GPUAcceleratedMonteCarlo:
    """GPU-accelerated multivariate Student-t Monte Carlo simulation — a
    drop-in accelerated version of `StressTester.student_t_monte_carlo`'s
    core simulation loop, useful when running millions of scenarios (e.g.
    for a full multi-asset, multi-horizon stress-test sweep) where the
    pure-NumPy path becomes the bottleneck.
    """

    def __init__(self, backend: str | None = None):
        self.backend = backend or best_backend()

    def simulate(self, mu: np.ndarray, cov: np.ndarray, df: float, n_sims: int,
                 seed: int = 42) -> np.ndarray:
        if self.backend == "torch":
            return self._simulate_torch(mu, cov, df, n_sims, seed)
        if self.backend == "jax":
            return self._simulate_jax(mu, cov, df, n_sims, seed)
        if self.backend == "cupy":
            return self._simulate_cupy(mu, cov, df, n_sims, seed)
        return self._simulate_numpy(mu, cov, df, n_sims, seed)

    @staticmethod
    def _simulate_numpy(mu, cov, df, n_sims, seed):
        rng = np.random.default_rng(seed)
        n = len(mu)
        scale = cov * (df - 2) / df
        eigval, eigvec = np.linalg.eigh((scale + scale.T) / 2)
        eigval = np.clip(eigval, 1e-12, None)
        scale = eigvec @ np.diag(eigval) @ eigvec.T
        g = rng.chisquare(df, size=n_sims) / df
        z = rng.multivariate_normal(np.zeros(n), scale, size=n_sims)
        return mu + z / np.sqrt(g)[:, None]

    @staticmethod
    def _simulate_torch(mu, cov, df, n_sims, seed):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device)
        cov_t = torch.as_tensor(cov, dtype=torch.float32, device=device)
        scale = cov_t * (df - 2) / df
        scale = (scale + scale.T) / 2
        # ensure PSD via eigen-clip on GPU
        eigval, eigvec = torch.linalg.eigh(scale)
        eigval = torch.clamp(eigval, min=1e-12)
        scale = eigvec @ torch.diag(eigval) @ eigvec.T

        mvn = torch.distributions.MultivariateNormal(torch.zeros_like(mu_t), covariance_matrix=scale)
        z = mvn.sample((n_sims,))
        chi2 = torch.distributions.Chi2(df)
        g = chi2.sample((n_sims,)).to(device) / df
        sims = mu_t + z / torch.sqrt(g).unsqueeze(-1)
        return sims.cpu().numpy()

    @staticmethod
    def _simulate_jax(mu, cov, df, n_sims, seed):
        import jax
        import jax.numpy as jnp
        key = jax.random.PRNGKey(seed)
        n = len(mu)
        scale = cov * (df - 2) / df
        scale = (scale + scale.T) / 2
        eigval, eigvec = jnp.linalg.eigh(scale)
        eigval = jnp.clip(eigval, 1e-12, None)
        scale = eigvec @ jnp.diag(eigval) @ eigvec.T

        key1, key2 = jax.random.split(key)
        z = jax.random.multivariate_normal(key1, jnp.zeros(n), scale, shape=(n_sims,))
        g = jax.random.chisquare(key2, df, shape=(n_sims,)) / df
        sims = mu + z / jnp.sqrt(g)[:, None]
        return np.array(sims)

    @staticmethod
    def _simulate_cupy(mu, cov, df, n_sims, seed):
        import cupy as cp
        cp.random.seed(seed)
        n = len(mu)
        mu_c, cov_c = cp.asarray(mu), cp.asarray(cov)
        scale = cov_c * (df - 2) / df
        scale = (scale + scale.T) / 2
        eigval, eigvec = cp.linalg.eigh(scale)
        eigval = cp.clip(eigval, 1e-12, None)
        scale = eigvec @ cp.diag(eigval) @ eigvec.T
        z = cp.random.multivariate_normal(cp.zeros(n), scale, size=n_sims)
        g = cp.random.chisquare(df, size=n_sims) / df
        sims = mu_c + z / cp.sqrt(g)[:, None]
        return cp.asnumpy(sims)


class GPUAcceleratedResampling:
    """GPU-accelerated batched covariance/mean estimation for Michaud
    resampling — computes all B resampled (mu, Sigma) pairs as one batched
    tensor operation instead of B sequential NumPy calls, which is where
    the wall-clock win actually shows up for large resample counts.
    """

    def __init__(self, backend: str | None = None):
        self.backend = backend or best_backend()

    def batch_mean_cov(self, resampled_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """resampled_returns: (B, T, N) array of B bootstrap samples.
        Returns (B, N) means and (B, N, N) covariances, computed as one
        batched operation on whichever backend is active.
        """
        if self.backend == "torch":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            X = torch.as_tensor(resampled_returns, dtype=torch.float32, device=device)
            mean = X.mean(dim=1)
            centered = X - mean.unsqueeze(1)
            cov = torch.einsum("btn,btm->bnm", centered, centered) / (X.shape[1] - 1)
            return mean.cpu().numpy(), cov.cpu().numpy()
        # NumPy (and a reasonable fallback for jax/cupy without extra code paths)
        mean = resampled_returns.mean(axis=1)
        centered = resampled_returns - mean[:, None, :]
        cov = np.einsum("btn,btm->bnm", centered, centered) / (resampled_returns.shape[1] - 1)
        return mean, cov
