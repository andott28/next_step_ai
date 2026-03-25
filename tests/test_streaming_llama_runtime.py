import json
from pathlib import Path

import torch

from llama3_neuroplastic.experiments import streaming_llama_runtime as runtime_mod


def _write_index(snapshot_dir: Path) -> None:
    payload = {
        "weight_map": {
            "model.layers.0.mlp.up_proj.weight": "model-00001-of-00044.safetensors",
            "model.layers.0.mlp.up_proj.weight.absmax": "model-00001-of-00044.safetensors",
            "model.layers.1.mlp.up_proj.weight": "model-00002-of-00044.safetensors",
            "model.layers.1.mlp.up_proj.weight.absmax": "model-00002-of-00044.safetensors",
        }
    }
    (snapshot_dir / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")


def test_loader_reopens_shards_when_cache_disabled(tmp_path, monkeypatch) -> None:
    _write_index(tmp_path)
    opened_paths = []

    class _FakeHandle:
        def __init__(self, path: str) -> None:
            self.path = path

        def __enter__(self):
            opened_paths.append(self.path)
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get_tensor(self, key: str) -> torch.Tensor:
            return torch.tensor([len(opened_paths)], dtype=torch.float32)

    monkeypatch.setattr(runtime_mod, "safe_open", lambda path, framework, device: _FakeHandle(path))
    loader = runtime_mod.ShardedSafetensorLoader(tmp_path, cache_shard_handles=False)

    first = loader._load_exact_tensors(["model.layers.0.mlp.up_proj.weight"])
    second = loader._load_exact_tensors(["model.layers.0.mlp.up_proj.weight"])

    assert len(opened_paths) == 2
    assert loader._shard_handles == {}
    assert first["model.layers.0.mlp.up_proj.weight"].item() == 1.0
    assert second["model.layers.0.mlp.up_proj.weight"].item() == 2.0


def test_loader_reuses_shards_when_cache_enabled(tmp_path, monkeypatch) -> None:
    _write_index(tmp_path)
    opened_paths = []

    class _FakeHandle:
        def __init__(self, path: str) -> None:
            self.path = path

        def __enter__(self):
            raise AssertionError("cached path should not use context-manager mode")

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get_tensor(self, key: str) -> torch.Tensor:
            return torch.tensor([1.0], dtype=torch.float32)

    def _fake_safe_open(path: str, framework: str, device: str):
        opened_paths.append(path)
        return _FakeHandle(path)

    monkeypatch.setattr(runtime_mod, "safe_open", _fake_safe_open)
    loader = runtime_mod.ShardedSafetensorLoader(tmp_path, cache_shard_handles=True)

    loader._load_exact_tensors(["model.layers.0.mlp.up_proj.weight"])
    loader._load_exact_tensors(["model.layers.0.mlp.up_proj.weight"])

    assert len(opened_paths) == 1
    assert set(loader._shard_handles.keys()) == {"model-00001-of-00044.safetensors"}
