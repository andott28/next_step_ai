import argparse
import subprocess
import sys


def run_once(neuroplastic_off: bool, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "llama3_neuroplastic/run_llama_benchmark_1000_added_buffer_interpolateinject.py",
        "--batch_size",
        str(args.batch_size),
        "--max_length",
        str(args.max_length),
    ]
    if args.max_samples is not None:
        cmd += ["--max_samples", str(args.max_samples)]
    if args.checkpoint_dir and not args.fresh:
        cmd += ["--checkpoint-dir", args.checkpoint_dir]
    if args.fresh:
        cmd.append("--fresh")
    if args.task_id is not None:
        cmd += ["--task-id", str(args.task_id)]
    if neuroplastic_off:
        cmd.append("--neuroplastic-off")

    print("\n" + "=" * 50)
    print("BASELINE" if neuroplastic_off else "NEUROPLASTIC")
    print("=" * 50)
    if args.fresh:
        print("source: fresh base model (no local checkpoint); random neuroplastic params disabled")
    elif not args.checkpoint_dir:
        print("source: no checkpoint provided (uses in-script defaults)")
    else:
        print(f"source: {args.checkpoint_dir}")
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and neuroplastic benchmark sequentially")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--task-id", type=int, default=None, help="Optional fixed task_id to pass through.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoint-dir and run as fresh base model without random untrained neuroplastic perturbations",
    )
    args = parser.parse_args()

    rc1 = run_once(neuroplastic_off=True, args=args)
    rc2 = run_once(neuroplastic_off=False, args=args)
    sys.exit(0 if (rc1 == 0 and rc2 == 0) else 1)


if __name__ == "__main__":
    main()
