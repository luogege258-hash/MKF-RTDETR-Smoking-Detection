"""Run numerical formula checks and paper-size module forwards without training."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.extra_modules.amkf import AMKF, AMKFBlock  # noqa: E402
from ultralytics.nn.extra_modules.epla import (  # noqa: E402
    EPLA,
    FrequencyEnhancement,
    PolarizedLinearAttention,
    TransformerEncoderLayer_EPLA,
)
from ultralytics.nn.extra_modules.esrfpn import BiFuse, ESRCA, ESRFPN  # noqa: E402


def require(condition: bool, message: str) -> None:
    """Raise even when Python is started with ``-O`` (unlike bare assert)."""
    if not condition:
        raise AssertionError(message)


def require_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape)
    require(actual == expected, f"{name}: expected shape {expected}, got {actual}")
    require(bool(torch.isfinite(tensor).all().item()), f"{name}: output contains NaN or Inf")


def require_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    require(
        bool(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)),
        f"{name}: numerical result does not match the manuscript formula; "
        f"max_abs_error={(actual - expected).abs().max().item():.6g}",
    )


def check_linear_attention(device: torch.device) -> None:
    """Equation (8): LinearAttn(Q,K,V) = Q(K^T V), without a denominator."""
    q = torch.tensor(
        [[[[1.0, -2.0], [0.5, 3.0], [-1.5, 2.0]]]], device=device
    )
    k = torch.tensor(
        [[[[2.0, 1.0], [-1.0, 0.5], [3.0, -2.0]]]], device=device
    )
    value = torch.tensor(
        [[[[0.25, 2.0], [1.5, -1.0], [-0.5, 4.0]]]], device=device
    )
    actual = PolarizedLinearAttention._linear_attention(q, k, value)
    expected = q @ (k.transpose(-2, -1) @ value)
    require_close(actual, expected, "LinearAttn = Q(K^T V)")


def check_frequency_identity(device: torch.device) -> None:
    """Equations (9)--(10): verify the zero gate and an all-pass filter."""
    module = FrequencyEnhancement(dim=16, frequency_length=625).to(device).eval()
    with torch.no_grad():
        module.freq_real.normal_()
        module.freq_imag.normal_()
        module.gamma.zero_()
    feature = torch.randn(2, 625, 16, device=device)
    actual = module(feature)
    require_close(actual, feature, "frequency branch identity at gamma=0")
    require(
        tuple(module.freq_real.shape) == (625 // 2 + 1, 16),
        "rFFT filter must have N//2+1 frequency bins",
    )

    # With H(f)=1, iFFT(FFT(F) * H)=F, so equation (10) also has a
    # closed-form reference when the learnable gate is non-zero.
    with torch.no_grad():
        module.freq_real.fill_(1)
        module.freq_imag.zero_()
        module.gamma.fill_(0.7)
    expected = feature * (1 + torch.tanh(module.gamma))
    require_close(module(feature), expected, "frequency all-pass equations (9)--(10)")


def check_bifuse(device: torch.device) -> None:
    """Equations (13)--(15): verify both gated paths and their sum."""
    module = BiFuse(main_channels=4, context_channels=6).to(device).eval()
    main = torch.randn(2, 4, 7, 5, device=device)
    context = torch.randn(2, 6, 3, 4, device=device)
    resized_context = F.interpolate(context, size=main.shape[-2:], mode="nearest")
    forward_path = main * torch.sigmoid(module.context_gate(resized_context))
    reverse_path = module.context_projection(resized_context) * torch.sigmoid(module.main_gate(main))
    expected = forward_path + reverse_path
    actual = module(main, context)
    require_close(actual, expected, "BiFuse equations (13)--(15)")


def check_epla_backward(device: torch.device) -> None:
    """Check train-mode gradients, with CUDA autocast when a GPU is available."""
    module = EPLA(32, num_heads=4, frequency_length=25).to(device).train()
    sequence = torch.randn(2, 25, 32, device=device, requires_grad=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        loss = module(sequence, 5, 5).float().square().mean()
    loss.backward()
    require(sequence.grad is not None, "EPLA backward did not produce an input gradient")
    require(bool(torch.isfinite(sequence.grad).all().item()), "EPLA input gradient contains NaN or Inf")
    parameter_gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    require(bool(parameter_gradients), "EPLA backward produced no parameter gradients")
    require(
        all(bool(torch.isfinite(gradient).all().item()) for gradient in parameter_gradients),
        "EPLA parameter gradient contains NaN or Inf",
    )


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Formula-check device: {device}")

    with torch.inference_mode():
        check_linear_attention(device)
        check_frequency_identity(device)
        check_bifuse(device)

        amkf = AMKFBlock(32).to(device).eval()
        amkf_input = torch.randn(1, 32, 16, 16, device=device)
        require_shape(amkf(amkf_input), (1, 32, 16, 16), "AMKFBlock")

        # The paper inserts the formula-defined transform into a C2f stage.
        amkf_c2f = AMKF(32, 64, n=2).to(device).eval()
        require_shape(amkf_c2f(amkf_input), (1, 64, 16, 16), "AMKF C2f wrapper")

        # N=625 and 25x25 are the manuscript's S5'/AIFI resolution at 800x800 input.
        epla = EPLA(64, num_heads=8, frequency_length=625).to(device).eval()
        sequence = torch.randn(1, 625, 64, device=device)
        require_shape(epla(sequence, 25, 25), (1, 625, 64), "EPLA N=625")
        require(
            epla.polar_attention.polar_projection.in_features == 4 * 64,
            "EPLA must concatenate four polarity paths before projection",
        )
        require(
            tuple(epla.frequency_enhancement.freq_real.shape) == (313, 64),
            "EPLA rFFT filter shape must be (625//2+1, C)",
        )

        # RT-DETR's YAML parser calls this adapter with a BCHW tensor.
        epla_adapter = TransformerEncoderLayer_EPLA(
            64, cm=256, num_heads=8, frequency_length=625
        ).to(device).eval()
        feature_map = torch.randn(1, 64, 25, 25, device=device)
        adapter_output = epla_adapter(feature_map)
        require_shape(
            adapter_output,
            (1, 64, 25, 25),
            "TransformerEncoderLayer_EPLA BCHW adapter",
        )
        direct_output = epla_adapter.epla(
            feature_map.flatten(2).transpose(1, 2), 25, 25
        ).transpose(1, 2).reshape_as(feature_map)
        require_close(
            adapter_output,
            direct_output,
            "EPLA adapter must not add residual/LN/FFN operations",
        )

        # Batch size 2 catches accidental broadcasting across the batch axis.
        esrca = ESRCA(32).to(device).eval()
        require_shape(
            esrca(torch.randn(2, 32, 8, 8, device=device)),
            (2, 32, 8, 8),
            "ESRCA",
        )

        # Exact paper pyramid sizes for an 800x800 input; batch 1 keeps memory bounded.
        fpn = ESRFPN((128, 256, 256), 256).to(device).eval()
        p3, p4, p5 = fpn(
            (
                torch.randn(1, 128, 100, 100, device=device),
                torch.randn(1, 256, 50, 50, device=device),
                torch.randn(1, 256, 25, 25, device=device),
            )
        )
        require_shape(p3, (1, 256, 100, 100), "ESRFPN P3")
        require_shape(p4, (1, 256, 50, 50), "ESRFPN P4")
        require_shape(p5, (1, 256, 25, 25), "ESRFPN P5")
        require(sum(fpn.in_channels) == 640, "Enhanced PCE input must total 640 channels")

    check_epla_backward(device)

    print("Formula-aligned AMKF/EPLA/ESRFPN numerical and shape checks passed")


if __name__ == "__main__":
    main()
