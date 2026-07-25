"""
tests/test_config_weights.py
可調整權重持久化（get_active_factor_weights / save_factor_weights）測試
"""

import json

import config


def test_default_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WEIGHTS_STORE_PATH", str(tmp_path / "nope.json"))
    assert config.get_active_factor_weights() == config.FACTOR_WEIGHTS


def test_save_and_get_roundtrip(tmp_path, monkeypatch):
    path = str(tmp_path / "weights.json")
    monkeypatch.setattr(config, "WEIGHTS_STORE_PATH", path)
    custom = {"chips": 0.25, "fundamental": 0.40, "technical": 0.10,
              "momentum": 0.10, "risk": 0.15}
    config.save_factor_weights(custom, {"updated_at": "2026-07-27"})
    got = config.get_active_factor_weights()
    assert got == custom
    # metadata 一併寫入
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["updated_at"] == "2026-07-27"


def test_invalid_missing_keys_falls_back(tmp_path, monkeypatch):
    path = str(tmp_path / "weights.json")
    monkeypatch.setattr(config, "WEIGHTS_STORE_PATH", path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"factor_weights": {"chips": 0.5}}, f)   # 缺鍵
    assert config.get_active_factor_weights() == config.FACTOR_WEIGHTS


def test_invalid_json_falls_back(tmp_path, monkeypatch):
    path = str(tmp_path / "weights.json")
    monkeypatch.setattr(config, "WEIGHTS_STORE_PATH", path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ not json")
    assert config.get_active_factor_weights() == config.FACTOR_WEIGHTS


def test_get_weights_meta(tmp_path, monkeypatch):
    path = str(tmp_path / "weights.json")
    monkeypatch.setattr(config, "WEIGHTS_STORE_PATH", path)
    assert config.get_weights_meta() == {}          # 無檔
    config.save_factor_weights(config.FACTOR_WEIGHTS,
                               {"updated_at": "2026-07-27", "win_rate_60d": 0.62, "samples": 30})
    meta = config.get_weights_meta()
    assert meta["updated_at"] == "2026-07-27"
    assert meta["win_rate_60d"] == 0.62
    assert meta["samples"] == 30
