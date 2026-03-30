import torch

from llama3_neuroplastic import performance_utils as perf_utils


def test_resolve_dataloader_kwargs_omits_worker_only_fields_when_single_process() -> None:
    kwargs = perf_utils.resolve_dataloader_kwargs(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    assert kwargs["batch_size"] == 2
    assert kwargs["num_workers"] == 0
    assert kwargs["pin_memory"] is False
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_resolve_dataloader_kwargs_auto_workers(monkeypatch) -> None:
    monkeypatch.setattr(perf_utils.os, "cpu_count", lambda: 10)

    kwargs = perf_utils.resolve_dataloader_kwargs(
        batch_size=1,
        num_workers=-1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=3,
    )

    assert kwargs["num_workers"] == 8
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 3


def test_build_optimizer_auto_falls_back_to_adamw_when_bitsandbytes_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(perf_utils, "bnb", None)
    param = torch.nn.Parameter(torch.ones(1))

    optimizer, metadata = perf_utils.build_optimizer(
        [param],
        optimizer_name="auto",
        lr=1e-3,
        weight_decay=0.0,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert metadata["uses_bitsandbytes"] is False


def test_resolve_amp_dtype_disables_amp_on_cpu() -> None:
    enabled, dtype = perf_utils.resolve_amp_dtype(device=torch.device("cpu"), requested="float16")

    assert enabled is False
    assert dtype == torch.float32
