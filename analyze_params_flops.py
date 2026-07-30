import argparse
import importlib
import inspect
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import reduce
from operator import mul
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ClipVitSpec:
    name: str
    image_resolution: int
    patch_size: int
    vision_width: int
    vision_layers: int
    embed_dim: int
    context_length: int
    vocab_size: int
    text_width: int
    text_layers: int


CLIP_VIT_SPECS = {
    "CLIP:ViT-L/14": ClipVitSpec(
        name="CLIP:ViT-L/14",
        image_resolution=224,
        patch_size=14,
        vision_width=1024,
        vision_layers=24,
        embed_dim=768,
        context_length=77,
        vocab_size=49408,
        text_width=768,
        text_layers=12,
    ),
    "CLIP:ViT-L/14@336px": ClipVitSpec(
        name="CLIP:ViT-L/14@336px",
        image_resolution=336,
        patch_size=14,
        vision_width=1024,
        vision_layers=24,
        embed_dim=768,
        context_length=77,
        vocab_size=49408,
        text_width=768,
        text_layers=12,
    ),
}


ABLATION_NAMES = {
    0: "SPARE dual-branch gated response",
    1: "origin branch only",
    2: "response branch only",
    3: "gated response branch only",
    4: "dual branch without gating",
    5: "Gaussian perturbation",
    6: "patch masking perturbation",
}


@dataclass
class ParamBreakdown:
    clip_total: int
    clip_visual: int
    clip_text_unused_in_image_forward: int
    spare_trainable: int
    image_forward_params: int
    registered_total: int


@dataclass
class FlopBreakdown:
    vision_passes: int
    clip_vision_macs_per_pass: int
    clip_vision_macs_total: int
    spare_head_macs: int
    total_macs: int
    total_flops: int


@dataclass
class Report:
    method: str
    arch: str
    ablation: int
    ablation_name: str
    input_size: Tuple[int, int]
    batch_size: int
    select_k: int
    k: int
    params: ParamBreakdown
    flops: FlopBreakdown
    notes: List[str]


def prod(values: Iterable[int]) -> int:
    return int(reduce(mul, values, 1))


def linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def layer_norm_params(width: int) -> int:
    return 2 * width


def transformer_block_params(width: int, mlp_ratio: int = 4) -> int:
    hidden = width * mlp_ratio
    attn = linear_params(width, 3 * width) + linear_params(width, width)
    mlp = linear_params(width, hidden) + linear_params(hidden, width)
    norms = 2 * layer_norm_params(width)
    return attn + mlp + norms


def clip_vit_params(spec: ClipVitSpec) -> Tuple[int, int, int]:
    grid = spec.image_resolution // spec.patch_size

    vision = 0
    vision += spec.vision_width * 3 * spec.patch_size * spec.patch_size
    vision += spec.vision_width
    vision += (grid * grid + 1) * spec.vision_width
    vision += layer_norm_params(spec.vision_width)
    vision += spec.vision_layers * transformer_block_params(spec.vision_width)
    vision += layer_norm_params(spec.vision_width)
    vision += spec.vision_width * spec.embed_dim

    text = 0
    text += spec.vocab_size * spec.text_width
    text += spec.context_length * spec.text_width
    text += spec.text_layers * transformer_block_params(spec.text_width)
    text += layer_norm_params(spec.text_width)
    text += spec.text_width * spec.embed_dim
    text += 1

    return vision + text, vision, text


def spare_layout(ablation: int, k: int) -> Dict[str, int]:
    if ablation == 1:
        return {
            "vision_passes": 1,
            "head_encoder_calls": 1,
            "branches": 1,
            "classifier_width_multiplier": 1,
            "uses_gating": 0,
        }

    if ablation in (2, 3):
        return {
            "vision_passes": 1 + k,
            "head_encoder_calls": 1,
            "branches": 1,
            "classifier_width_multiplier": 1,
            "uses_gating": int(ablation == 3),
        }

    return {
        "vision_passes": 1 + k,
        "head_encoder_calls": 2,
        "branches": 2,
        "classifier_width_multiplier": 2,
        "uses_gating": int(ablation in (0, 5, 6)),
    }


def spare_trainable_params(select_k: int, ablation: int, width: int = 1024) -> int:
    layout = spare_layout(ablation, k=1)
    num_hooked_layers = 9
    num_choices = num_hooked_layers - select_k + 1
    if num_choices <= 0:
        raise ValueError(
            f"select_k ({select_k}) must be in [1, {num_hooked_layers}] for this SPARE implementation."
        )

    selector = num_choices
    branch_tokens = layout["branches"] * (select_k * width + width)
    gating = select_k * (linear_params(4, 16) + linear_params(16, 1)) if layout["uses_gating"] else 0
    encoder = transformer_block_params(width)
    classifier_in = layout["classifier_width_multiplier"] * width
    classifier = layer_norm_params(classifier_in) + linear_params(classifier_in, 1)

    return selector + branch_tokens + gating + encoder + classifier


def transformer_block_macs(batch_size: int, sequence_length: int, width: int, mlp_ratio: int = 4) -> int:
    hidden = width * mlp_ratio
    qkv = batch_size * sequence_length * width * (3 * width)
    out_proj = batch_size * sequence_length * width * width
    attention_scores = batch_size * sequence_length * sequence_length * width
    attention_values = batch_size * sequence_length * sequence_length * width
    mlp = batch_size * sequence_length * width * hidden
    mlp += batch_size * sequence_length * hidden * width
    return qkv + out_proj + attention_scores + attention_values + mlp


def clip_vit_macs_per_pass(spec: ClipVitSpec, batch_size: int, input_size: Tuple[int, int]) -> int:
    height, width = input_size
    if height % spec.patch_size != 0 or width % spec.patch_size != 0:
        raise ValueError(
            f"input size {height}x{width} must be divisible by CLIP patch size {spec.patch_size}."
        )

    grid_h = height // spec.patch_size
    grid_w = width // spec.patch_size
    sequence_length = grid_h * grid_w + 1

    conv = batch_size * spec.vision_width * grid_h * grid_w * 3 * spec.patch_size * spec.patch_size
    blocks = spec.vision_layers * transformer_block_macs(batch_size, sequence_length, spec.vision_width)
    projection = batch_size * spec.vision_width * spec.embed_dim
    return conv + blocks + projection


def spare_head_macs(batch_size: int, select_k: int, ablation: int, width: int = 1024) -> int:
    layout = spare_layout(ablation, k=1)
    sequence_length = select_k + 1
    head = layout["head_encoder_calls"] * transformer_block_macs(batch_size, sequence_length, width)
    if layout["uses_gating"]:
        head += batch_size * select_k * (4 * 16 + 16 * 1)
    classifier_in = layout["classifier_width_multiplier"] * width
    head += batch_size * classifier_in
    return head


def normalize_arch(arch: str) -> str:
    if arch in CLIP_VIT_SPECS:
        return arch
    if not arch.startswith("CLIP:") and f"CLIP:{arch}" in CLIP_VIT_SPECS:
        return f"CLIP:{arch}"
    return arch


def analytic_report(args: argparse.Namespace, k: int) -> Report:
    arch = normalize_arch(args.arch)
    if arch not in CLIP_VIT_SPECS:
        supported = ", ".join(sorted(CLIP_VIT_SPECS))
        raise ValueError(f"analytic mode currently supports {supported}; use --method runtime for other models.")

    spec = CLIP_VIT_SPECS[arch]
    input_size = parse_input_size(args.input_size, default=spec.image_resolution)
    clip_total, clip_visual, clip_text = clip_vit_params(spec)
    trainable = spare_trainable_params(args.select_k, args.ablation, spec.vision_width)
    layout = spare_layout(args.ablation, k)
    per_pass = clip_vit_macs_per_pass(spec, args.batch_size, input_size)
    vision_total = layout["vision_passes"] * per_pass
    head = spare_head_macs(args.batch_size, args.select_k, args.ablation, spec.vision_width)
    total_macs = vision_total + head

    notes = [
        "FLOPs are reported as 2 * MACs.",
        "Patch shuffle/mask/noise indexing, LayerNorm arithmetic, GELU, softmax, and tensor adds are not included.",
        "CLIP text-encoder parameters are registered in the loaded CLIP module but are not used by image-only inference.",
    ]
    if input_size != (spec.image_resolution, spec.image_resolution):
        notes.append(
            f"{arch} was pretrained for {spec.image_resolution}x{spec.image_resolution}; "
            "the formula can scale to this size, but the repository model may need positional-embedding handling to run it."
        )

    return Report(
        method="analytic",
        arch=arch,
        ablation=args.ablation,
        ablation_name=ABLATION_NAMES[args.ablation],
        input_size=input_size,
        batch_size=args.batch_size,
        select_k=args.select_k,
        k=k,
        params=ParamBreakdown(
            clip_total=clip_total,
            clip_visual=clip_visual,
            clip_text_unused_in_image_forward=clip_text,
            spare_trainable=trainable,
            image_forward_params=clip_visual + trainable,
            registered_total=clip_total + trainable,
        ),
        flops=FlopBreakdown(
            vision_passes=layout["vision_passes"],
            clip_vision_macs_per_pass=per_pass,
            clip_vision_macs_total=vision_total,
            spare_head_macs=head,
            total_macs=total_macs,
            total_flops=2 * total_macs,
        ),
        notes=notes,
    )


def configure_clip_download_root(download_root: Optional[str]) -> None:
    if not download_root:
        return

    import models.clip as clip_package
    import models.clip.clip as clip_module

    original_package_load = clip_package.load
    original_module_load = clip_module.load

    def package_load(name, device="cuda", jit=False, download_root=None):
        return original_package_load(name, device=device, jit=jit, download_root=download_root or download_root_arg)

    def module_load(name, device="cuda", jit=False, download_root=None):
        return original_module_load(name, device=device, jit=jit, download_root=download_root or download_root_arg)

    download_root_arg = download_root
    clip_package.load = package_load
    clip_module.load = module_load


def build_runtime_model(args: argparse.Namespace):
    configure_clip_download_root(args.download_root)

    if args.ablation == 0:
        module = importlib.import_module("models.clip_models")
    else:
        module = importlib.import_module(f"models.ablation.ablation{args.ablation}")

    clip_name = args.arch[5:] if args.arch.startswith("CLIP:") else args.arch
    kwargs = {
        "num_classes": 1,
        "select_num": args.select_k,
        "training": False,
        "p": args.p,
    }
    signature = inspect.signature(module.CLIPModel.__init__)
    if "k" in signature.parameters:
        kwargs["k"] = args.k
    elif args.k != 1:
        print(
            f"warning: {module.__name__}.CLIPModel does not accept k; runtime profile uses one perturbed view."
        )

    return module.CLIPModel(clip_name, **kwargs)


def tensor_from(value):
    try:
        import torch
    except ImportError:
        return None

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = tensor_from(item)
            if tensor is not None:
                return tensor
    return None


def runtime_report(args: argparse.Namespace, k: int) -> Report:
    if k != args.k:
        raise ValueError("runtime mode profiles one k value at a time; omit --k-values.")

    import torch
    import torch.nn as nn

    input_size = parse_input_size(args.input_size, default=224)
    model = build_runtime_model(args)
    device = torch.device(args.device)
    model.eval().to(device)

    stats = defaultdict(lambda: {"macs": 0, "calls": 0})
    handles = []

    def add(name: str, kind: str, macs: int) -> None:
        key = f"{kind}:{name}"
        stats[key]["macs"] += int(macs)
        stats[key]["calls"] += 1

    def conv_hook(name: str, module, inputs, output) -> None:
        out = tensor_from(output)
        if out is None or out.dim() != 4:
            return
        batch, out_channels, out_h, out_w = out.shape
        kernel_h, kernel_w = module.kernel_size
        macs = batch * out_channels * out_h * out_w
        macs *= (module.in_channels // module.groups) * kernel_h * kernel_w
        add(name, "conv2d", macs)

    def linear_hook(name: str, module, inputs, output) -> None:
        out = tensor_from(output)
        if out is None:
            return
        macs = prod(out.shape) * module.in_features
        add(name, "linear", macs)

    def mha_hook(name: str, module, inputs, output) -> None:
        query = tensor_from(inputs[0])
        key = tensor_from(inputs[1]) if len(inputs) > 1 else query
        if query is None or key is None:
            return
        if getattr(module, "batch_first", False):
            batch = query.shape[0]
            q_len = query.shape[1]
            k_len = key.shape[1]
        else:
            q_len = query.shape[0]
            batch = query.shape[1]
            k_len = key.shape[0]
        embed_dim = module.embed_dim
        qkv = batch * (q_len + 2 * k_len) * embed_dim * embed_dim
        attn = 2 * batch * q_len * k_len * embed_dim
        out_proj = batch * q_len * embed_dim * embed_dim
        add(name, "multihead_attention", qkv + attn + out_proj)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(lambda m, i, o, n=name: conv_hook(n, m, i, o)))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(lambda m, i, o, n=name: mha_hook(n, m, i, o)))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(lambda m, i, o, n=name: linear_hook(n, m, i, o)))

    height, width = input_size
    dummy = torch.randn(args.batch_size, 3, height, width, device=device)
    with torch.no_grad():
        model(dummy)

    for handle in handles:
        handle.remove()

    total_macs = sum(item["macs"] for item in stats.values())
    vision_passes = stats.get("conv2d:model.visual.conv1", {}).get("calls", 0)
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is not None and getattr(visual, "proj", None) is not None and vision_passes:
        proj = visual.proj
        proj_macs = args.batch_size * int(proj.shape[0]) * int(proj.shape[1]) * vision_passes
        total_macs += proj_macs
        stats["matmul:model.visual.proj"]["macs"] += proj_macs
        stats["matmul:model.visual.proj"]["calls"] += vision_passes

    clip_total = sum(parameter.numel() for parameter in model.model.parameters()) if hasattr(model, "model") else 0
    clip_visual = sum(parameter.numel() for parameter in model.model.visual.parameters()) if hasattr(model, "model") else 0
    adapter = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("model.")
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    notes = [
        "Runtime MACs are counted with hooks for Conv2d, Linear, and MultiheadAttention.",
        "LayerNorm arithmetic, GELU, softmax, tensor adds, and patch perturbation indexing are not included.",
        "The visual projection matmul is added manually because the CLIP implementation uses the @ operator.",
    ]
    if adapter != trainable:
        notes.append("Trainable parameter count differs from adapter count; check requires_grad flags in the model.")

    return Report(
        method="runtime",
        arch=args.arch,
        ablation=args.ablation,
        ablation_name=ABLATION_NAMES[args.ablation],
        input_size=input_size,
        batch_size=args.batch_size,
        select_k=args.select_k,
        k=args.k,
        params=ParamBreakdown(
            clip_total=clip_total,
            clip_visual=clip_visual,
            clip_text_unused_in_image_forward=clip_total - clip_visual,
            spare_trainable=trainable,
            image_forward_params=clip_visual + adapter,
            registered_total=clip_total + adapter,
        ),
        flops=FlopBreakdown(
            vision_passes=vision_passes,
            clip_vision_macs_per_pass=0,
            clip_vision_macs_total=0,
            spare_head_macs=0,
            total_macs=total_macs,
            total_flops=2 * total_macs,
        ),
        notes=notes,
    )


def parse_input_size(values: Optional[List[int]], default: int) -> Tuple[int, int]:
    if values is None:
        return default, default
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("--input-size accepts either one integer or two integers.")


def compact_number(value: int, unit: str) -> str:
    scale = 1_000_000 if unit == "M" else 1_000_000_000
    return f"{value / scale:.3f}{unit}"


def pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0.00%"
    return f"{100.0 * part / whole:.2f}%"


def print_single_report(report: Report) -> None:
    params = report.params
    flops = report.flops
    height, width = report.input_size

    print(f"Method: {report.method}")
    print(f"Model: {report.arch} | {report.ablation_name} (ablation={report.ablation})")
    print(f"Input: batch={report.batch_size}, size={height}x{width}, select_k={report.select_k}, k={report.k}")
    print("")
    print("Parameters")
    print(f"  Registered total:       {params.registered_total:,} ({compact_number(params.registered_total, 'M')})")
    print(f"  Image-forward params:   {params.image_forward_params:,} ({compact_number(params.image_forward_params, 'M')})")
    print(f"  Trainable SPARE params: {params.spare_trainable:,} ({compact_number(params.spare_trainable, 'M')})")
    print(f"  Frozen CLIP visual:     {params.clip_visual:,} ({compact_number(params.clip_visual, 'M')})")
    print(f"  CLIP text unused:       {params.clip_text_unused_in_image_forward:,} ({compact_number(params.clip_text_unused_in_image_forward, 'M')})")
    print("")
    print("Compute")
    print(f"  Vision encoder passes:  {flops.vision_passes}")
    if flops.clip_vision_macs_per_pass:
        print(f"  CLIP vision / pass:     {compact_number(flops.clip_vision_macs_per_pass, 'G')} MACs")
        print(f"  CLIP vision total:      {compact_number(flops.clip_vision_macs_total, 'G')} MACs")
        print(f"  SPARE head:             {compact_number(flops.spare_head_macs, 'G')} MACs ({pct(flops.spare_head_macs, flops.total_macs)})")
    print(f"  Total:                  {compact_number(flops.total_macs, 'G')} MACs")
    print(f"  Total FLOPs:            {compact_number(flops.total_flops, 'G')} FLOPs")
    print("")
    print("Notes")
    for note in report.notes:
        print(f"  - {note}")


def print_table(reports: List[Report]) -> None:
    print("k,vision_passes,trainable_params,total_params,image_forward_params,GMACs,GFLOPs")
    for report in reports:
        print(
            f"{report.k},"
            f"{report.flops.vision_passes},"
            f"{report.params.spare_trainable},"
            f"{report.params.registered_total},"
            f"{report.params.image_forward_params},"
            f"{report.flops.total_macs / 1_000_000_000:.3f},"
            f"{report.flops.total_flops / 1_000_000_000:.3f}"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze SPARE parameter counts and image-forward MACs/FLOPs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch", default="CLIP:ViT-L/14")
    parser.add_argument("--ablation", type=int, default=0, choices=sorted(ABLATION_NAMES))
    parser.add_argument("--input-size", nargs="+", type=int, default=None, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--select_k", "--select-k", dest="select_k", type=int, default=5)
    parser.add_argument("--k", type=int, default=1, help="Number of independent perturbed views.")
    parser.add_argument("--k-values", nargs="+", type=int, default=None, help="Print a compact table for several k values.")
    parser.add_argument("--p", type=float, default=0.7, help="Perturbation ratio; it does not change MACs in the analytic formula.")
    parser.add_argument("--method", choices=("analytic", "runtime"), default="analytic")
    parser.add_argument("--device", default="cpu", help="Device used by --method runtime.")
    parser.add_argument("--download-root", default=None, help="Optional CLIP checkpoint cache for --method runtime.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be greater than or equal to 1.")
    if args.k_values and args.method == "runtime":
        raise ValueError("--k-values is only supported with --method analytic.")

    k_values = args.k_values or [args.k]
    reports = [
        analytic_report(args, k) if args.method == "analytic" else runtime_report(args, k)
        for k in k_values
    ]

    if args.json:
        payload = [asdict(report) for report in reports]
        print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2))
    elif len(reports) == 1:
        print_single_report(reports[0])
    else:
        print_table(reports)


if __name__ == "__main__":
    main()
