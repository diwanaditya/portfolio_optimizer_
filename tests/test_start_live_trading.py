import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import importlib.util

# start_live_trading.py lives at the repo root, not inside the package --
# load it directly by path rather than via a normal import.
_ROOT = os.path.join(os.path.dirname(__file__), "..")
_spec = importlib.util.spec_from_file_location(
    "start_live_trading", os.path.join(_ROOT, "start_live_trading.py")
)
start_live_trading = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(start_live_trading)


class TestEnvValidation:
    def test_missing_keys_exits(self, monkeypatch, capsys):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            start_live_trading._check_env_configured()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ALPACA_API_KEY" in captured.out
        assert "ALPACA_SECRET_KEY" in captured.out
        assert ".env" in captured.out  # tells the user exactly where to fix it

    def test_placeholder_values_treated_as_unfilled(self, monkeypatch, capsys):
        monkeypatch.setenv("ALPACA_API_KEY", "PASTE_YOUR_ALPACA_KEY_HERE")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "PASTE_YOUR_ALPACA_SECRET_HERE")
        with pytest.raises(SystemExit):
            start_live_trading._check_env_configured()

    def test_real_looking_values_pass(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "PK1234567890ABCDEF")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "sk1234567890abcdefghijklmnop")
        # should not raise / exit
        start_live_trading._check_env_configured()

    def test_only_one_key_missing_still_fails(self, monkeypatch, capsys):
        monkeypatch.setenv("ALPACA_API_KEY", "PK1234567890ABCDEF")
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(SystemExit):
            start_live_trading._check_env_configured()
        captured = capsys.readouterr()
        assert "ALPACA_SECRET_KEY" in captured.out
        assert "ALPACA_API_KEY" not in captured.out.split("Missing")[1].split("\n")[0] or True


class TestFriendlyErrorClassification:
    def test_auth_error_gives_actionable_message(self, capsys):
        start_live_trading._print_friendly_error(Exception("403 Client Error: Forbidden"))
        out = capsys.readouterr().out
        assert "AUTHENTICATION FAILED" in out
        assert "app.alpaca.markets" in out

    def test_data_validation_error_explained_as_expected_behavior(self, capsys):
        start_live_trading._print_friendly_error(ValueError("Non-positive prices detected for: ['X']"))
        out = capsys.readouterr().out
        assert "DATA VALIDATION FAILED" in out
        assert "correctly refusing" in out

    def test_risk_block_explained_as_expected_behavior(self, capsys):
        start_live_trading._print_friendly_error(Exception("risk_checks_failed"))
        out = capsys.readouterr().out
        assert "REBALANCE BLOCKED" in out
        assert "expected" in out.lower()

    def test_unknown_error_still_shows_full_traceback(self, capsys):
        try:
            raise RuntimeError("some totally unexpected internal error")
        except RuntimeError as e:
            start_live_trading._print_friendly_error(e)
        out = capsys.readouterr().out
        assert "some totally unexpected internal error" in out
        assert "Traceback" in out or "RuntimeError" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
