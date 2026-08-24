import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import builtins
import importlib
import pytest


def test_rl_agents_module_imports_without_torch(monkeypatch):
    """The critical regression test: simulate torch being unavailable and
    verify `import portfolio_optimizer.rl.agents` still succeeds (rather
    than crashing the whole process for anyone who imports this submodule
    without torch installed), and that instantiating an agent raises a
    clear, actionable ImportError instead of a confusing AttributeError.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated: torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Force a fresh import so the module-level try/except actually re-runs
    # under the simulated missing-torch condition.
    for mod_name in list(sys.modules):
        if mod_name.startswith("portfolio_optimizer.rl"):
            del sys.modules[mod_name]

    import portfolio_optimizer.rl.agents as agents
    assert agents._TORCH_AVAILABLE is False

    with pytest.raises(ImportError, match="require PyTorch"):
        agents.PPOAgent(state_dim=10, action_dim=4)
    with pytest.raises(ImportError, match="require PyTorch"):
        agents.SACAgent(state_dim=10, action_dim=4)
    with pytest.raises(ImportError, match="require PyTorch"):
        agents.DQNAgent(state_dim=10, n_assets=4)

    # Clean up: force re-import with torch available again for any other
    # test module that imports portfolio_optimizer.rl.agents after this one.
    monkeypatch.undo()
    for mod_name in list(sys.modules):
        if mod_name.startswith("portfolio_optimizer.rl"):
            del sys.modules[mod_name]
    importlib.import_module("portfolio_optimizer.rl.agents")


def test_gpu_accel_module_imports_without_optional_backends(monkeypatch):
    """gpu/accel.py already lazy-imports torch/jax/cupy inside functions
    (never at module level) -- verify that design holds: the module must
    import cleanly even when none of the optional GPU backends are
    installed, since NumPy fallback is supposed to always work.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("torch", "jax", "cupy") or name.startswith(("torch.", "jax.", "cupy.")):
            raise ImportError(f"simulated: {name} not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for mod_name in list(sys.modules):
        if mod_name.startswith("portfolio_optimizer.gpu"):
            del sys.modules[mod_name]

    import portfolio_optimizer.gpu.accel as accel
    backends = accel.available_backends()
    assert backends["torch"] is False
    assert backends["jax"] is False
    assert backends["cupy"] is False
    assert accel.best_backend() == "numpy"

    import numpy as np
    mc = accel.GPUAcceleratedMonteCarlo(backend="numpy")
    sims = mc.simulate(np.array([0.001, 0.002]), np.eye(2) * 0.0001, df=6, n_sims=100)
    assert sims.shape == (100, 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
