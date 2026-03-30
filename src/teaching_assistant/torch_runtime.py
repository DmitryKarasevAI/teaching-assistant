from __future__ import annotations

from typing import Any

import torch

from teaching_assistant.config_schema import TorchConfig


def resolve_required_cuda_device(cfg: TorchConfig) -> torch.device:
    device_text = (cfg.device or "").strip()

    if not cfg.require_cuda:
        raise RuntimeError(
            "CPU execution is disabled for this project. "
            "Set torch.require_cuda=true and use a CUDA device such as 'cuda:0'."
        )

    if not device_text:
        raise RuntimeError("torch.device must be set, e.g. 'cuda:0'.")

    if not device_text.startswith("cuda"):
        raise RuntimeError(
            f"Invalid torch.device={device_text!r}. "
            "Only CUDA devices are allowed, e.g. 'cuda:0'."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but this project requires every local torch model "
            "to run on GPU. Check the NVIDIA driver, Docker GPU passthrough, and the "
            "installed PyTorch CUDA wheel."
        )

    requested = torch.device(device_text)
    device_index = 0 if requested.index is None else int(requested.index)

    visible_device_count = torch.cuda.device_count()
    if device_index >= visible_device_count:
        raise RuntimeError(
            f"Configured torch.device={device_text!r}, but only "
            f"{visible_device_count} CUDA device(s) are visible."
        )

    torch.cuda.set_device(device_index)
    torch.empty(
        1, device=f"cuda:{device_index}"
    )  # fail fast if CUDA runtime wiring is broken
    return torch.device(f"cuda:{device_index}")


def single_gpu_device_map(device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got {device}.")
    return 0 if device.index is None else int(device.index)


def _unwrap_torch_module(obj: Any) -> Any | None:
    candidates = [
        obj,
        getattr(obj, "_model", None),
        getattr(obj, "model", None),
        getattr(getattr(obj, "_model", None), "model", None),
        getattr(getattr(obj, "model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "parameters"):
            return candidate
    return None


def assert_module_on_cuda(
    obj: Any, *, label: str, expected_device: torch.device
) -> None:
    module = _unwrap_torch_module(obj)
    if module is None:
        raise RuntimeError(
            f"{label} does not expose a torch module that can be device-validated."
        )

    tensors = list(module.parameters())
    if hasattr(module, "buffers"):
        tensors.extend(list(module.buffers()))

    if not tensors:
        raise RuntimeError(f"{label} has no parameters or buffers to validate.")

    expected_index = 0 if expected_device.index is None else int(expected_device.index)
    observed_devices = {str(tensor.device) for tensor in tensors}
    invalid_devices = sorted(
        {
            str(tensor.device)
            for tensor in tensors
            if tensor.device.type != "cuda"
            or int(tensor.device.index or 0) != expected_index
        }
    )

    if invalid_devices:
        raise RuntimeError(
            f"{label} is not fully on {expected_device}. "
            f"Observed parameter/buffer devices: {sorted(observed_devices)}"
        )
