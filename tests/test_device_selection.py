import torch

from trainer.device import describe_torch_device, resolve_torch_device


def _set_device_availability(monkeypatch, *, cuda: bool, mps: bool, mps_present: bool = True) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if mps_present:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    else:
        monkeypatch.delattr(torch.backends, "mps", raising=False)


def test_resolve_torch_device_prefers_cuda_over_mps(monkeypatch) -> None:
    _set_device_availability(monkeypatch, cuda=True, mps=True)

    device = resolve_torch_device()

    assert device.type == "cuda"
    assert describe_torch_device(device) == "cuda"


def test_resolve_torch_device_prefers_mps_when_cuda_is_unavailable(monkeypatch) -> None:
    _set_device_availability(monkeypatch, cuda=False, mps=True)

    device = resolve_torch_device()

    assert device.type == "mps"
    assert describe_torch_device(device) == "mps"


def test_resolve_torch_device_falls_back_to_cpu_when_mps_is_unavailable(monkeypatch) -> None:
    _set_device_availability(monkeypatch, cuda=False, mps=False)

    device = resolve_torch_device()

    assert device.type == "cpu"
    assert describe_torch_device(device) == "cpu"


def test_resolve_torch_device_falls_back_to_cpu_when_mps_backend_is_missing(monkeypatch) -> None:
    _set_device_availability(monkeypatch, cuda=False, mps=False, mps_present=False)

    device = resolve_torch_device()

    assert device.type == "cpu"
    assert describe_torch_device(device) == "cpu"
