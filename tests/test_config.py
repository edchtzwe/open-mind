"""config reload semantics: mtime-memo reads and the malformed-file trace.

The one behaviour worth pinning here is RULES.md A09: a malformed config.toml
must not be swallowed silently — it keeps the last good copy AND logs once.
"""

from __future__ import annotations

import logging

from openmind import config


def test_malformed_config_keeps_last_good_copy_and_logs(monkeypatch, tmp_path, caplog):
    good = tmp_path / "config.toml"
    good.write_text('[server]\nport = 8765\n', encoding="utf-8")
    monkeypatch.setattr(config, "_path", good)

    assert config.get("server", "port") == 8765  # cache the good copy

    good.write_text('[server\nport = broken toml', encoding="utf-8")  # break it
    with caplog.at_level(logging.WARNING, logger="openmind.config"):
        value = config.get("server", "port")

    assert value == 8765  # last good copy survives
    assert any("malformed" in record.getMessage() for record in caplog.records)


def test_missing_config_falls_back_to_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_path", tmp_path / "nope.toml")
    assert config.get("server", "host") == "127.0.0.1"
