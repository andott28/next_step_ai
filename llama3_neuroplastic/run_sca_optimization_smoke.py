import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], workdir: Path) -> None:
    print(f"[opt-smoke] running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(workdir), check=True)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_recal_summary(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    strict = data.get("strict_validation", {})
    quality = strict.get("quality", {})
    balance = strict.get("balance", {})
    return {
        "quality_gate_passed": bool(data.get("quality_gate_passed", False)),
        "initial_loss": data.get("initial_loss"),
        "min_loss": data.get("min_loss"),
        "final_loss": data.get("final_loss"),
        "strict_dense_top1_rank_median": strict.get("dense_top1_rank_median"),
        "strict_dense_top1_rank_p95": strict.get("dense_top1_rank_p95"),
        "strict_rollout_kl_mean": strict.get("rollout_kl_mean"),
        "strict_final_hidden_cosine_mean": strict.get("final_hidden_cosine_mean"),
        "strict_degenerate_frac": quality.get("degenerate_frac"),
        "max_block_importance_fraction": balance.get("max_block_importance_fraction"),
        "max_block_load_fraction": balance.get("max_block_load_fraction"),
    }


def _extract_probe_summary(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    rows = []
    for row in data.get("rows", []):
        prompts = row.get("prompts", [])
        rows.append(
            {
                "decode_guard_layers": row.get("decode_guard_layers"),
                "mean_deg": row.get("mean_deg"),
                "mean_rep": row.get("mean_rep"),
                "text": prompts[0].get("text", "") if prompts else "",
            }
        )
    return {"rows": rows}


def _build_recal_cmd(args: argparse.Namespace, output_dir: Path, *, steps: int, lr: float) -> List[str]:
    cmd = [
        sys.executable,
        str(Path("llama3_neuroplastic") / "run_sca_recalibration_from_hybrid_baseline.py"),
        "--model-name",
        args.model_name,
        "--hybrid-checkpoint",
        args.hybrid_checkpoint,
        "--learned-basis-init-checkpoint",
        args.learned_basis_init_checkpoint,
        "--output-dir",
        str(output_dir),
        "--recalibration-mode",
        "decode_manifold_alignment",
        "--layers",
        args.layers,
        "--sca-routing-mode",
        "semantic_latent",
        "--semantic-block-score-normalized" if args.semantic_block_score_normalized else "--no-semantic-block-score-normalized",
        "--sca-bottom-buffer-layers",
        str(args.sca_bottom_buffer_layers),
        "--sca-decode-guard-layers",
        str(args.sca_decode_guard_layers),
        "--basis-rank",
        str(args.basis_rank),
        "--basis-top-k",
        str(args.basis_top_k),
        "--top-k",
        str(args.top_k),
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--max-samples",
        str(args.max_samples),
        "--max-seq-length",
        str(args.max_seq_length),
        "--validation-prefix-count",
        str(args.validation_prefix_count),
        "--validation-prefix-length",
        str(args.validation_prefix_length),
        "--progressive-depth-enabled",
        "--progressive-depth-group-size",
        str(args.progressive_depth_group_size),
        "--no-include-spatial-proj",
        "--no-strict-decode-upper-layer-cap-enabled",
    ]
    return cmd


def _build_probe_cmd(args: argparse.Namespace, recal_state: Path, output_json: Path) -> List[str]:
    return [
        sys.executable,
        str(Path("llama3_neuroplastic") / "run_sca_fast_depth_probe.py"),
        "--checkpoint",
        args.hybrid_checkpoint,
        "--sca-recalibrated-checkpoint",
        str(recal_state),
        "--decode-guards",
        args.probe_decode_guards,
        "--sca-bottom-buffer-layers",
        str(args.sca_bottom_buffer_layers),
        "--sca-basis-rank",
        str(args.basis_rank),
        "--sca-basis-top-k",
        str(args.basis_top_k),
        "--sca-top-k",
        str(args.top_k),
        "--semantic-block-score-normalized" if args.semantic_block_score_normalized else "--no-semantic-block-score-normalized",
        "--allow-cache",
        "--max-new-tokens",
        str(args.probe_max_new_tokens),
        "--prompt",
        args.prompt,
        "--output-json",
        str(output_json),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Fast optimization-quality sweep for progressive sparse recalibration.")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--hybrid-checkpoint", type=str, required=True)
    p.add_argument("--learned-basis-init-checkpoint", type=str, required=True)
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--layers", type=str, default="2-5")
    p.add_argument("--sca-bottom-buffer-layers", type=int, default=2)
    p.add_argument("--sca-decode-guard-layers", type=int, default=26)
    p.add_argument("--semantic-block-score-normalized", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--basis-rank", type=int, default=96)
    p.add_argument("--basis-top-k", type=int, default=12)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--steps-list", type=str, default="8,16,32")
    p.add_argument("--lr-list", type=str, default="1e-5,3e-6")
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=64)
    p.add_argument("--validation-prefix-count", type=int, default=2)
    p.add_argument("--validation-prefix-length", type=int, default=48)
    p.add_argument("--progressive-depth-group-size", type=int, default=2)
    p.add_argument("--probe-decode-guards", type=str, default="26")
    p.add_argument("--probe-max-new-tokens", type=int, default=8)
    p.add_argument("--prompt", type=str, default="Write two clear factual sentences about Oslo.")
    args = p.parse_args()

    workdir = Path.cwd()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    steps_list = [int(v.strip()) for v in str(args.steps_list).split(",") if v.strip()]
    lr_list = [float(v.strip()) for v in str(args.lr_list).split(",") if v.strip()]
    results: List[Dict[str, Any]] = []

    for steps in steps_list:
        for lr in lr_list:
            tag = f"steps_{steps}_lr_{str(lr).replace('.', 'p').replace('-', 'm')}"
            run_dir = output_root / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            _run(_build_recal_cmd(args, run_dir, steps=steps, lr=lr), workdir)
            recal_metrics = run_dir / "sca_recalibration_metrics.json"
            recal_state = run_dir / "sca_recalibrated_state.pt"
            probe_json = run_dir / "fast_depth_probe.json"
            _run(_build_probe_cmd(args, recal_state, probe_json), workdir)
            results.append(
                {
                    "steps": steps,
                    "lr": lr,
                    "output_dir": str(run_dir),
                    "recalibration": _extract_recal_summary(recal_metrics),
                    "probe": _extract_probe_summary(probe_json),
                }
            )

    summary = {"config": vars(args), "runs": results}
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[opt-smoke] summary={summary_path}")


if __name__ == "__main__":
    main()
