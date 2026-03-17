from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache"}


@dataclass(frozen=True)
class Classification:
    category: str
    used: str
    reason: str


def _iter_files(root: Path, include_venv: bool) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if any(part in EXCLUDED_DIRS for part in parts):
            continue
        if not include_venv and parts and parts[0] == "verification_env":
            continue
        yield rel


def _module_name_from_relpath(rel: Path) -> Optional[str]:
    if rel.suffix != ".py":
        return None
    parts = list(rel.with_suffix("").parts)
    if not parts:
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def _build_module_map(py_files: Iterable[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for rel in py_files:
        mod = _module_name_from_relpath(rel)
        if mod:
            out[mod] = rel
    return out


def _resolve_import(
    *,
    cur_module: str,
    module: Optional[str],
    level: int,
) -> Optional[str]:
    if level == 0:
        return module
    cur_parts = cur_module.split(".")
    if cur_parts:
        cur_parts = cur_parts[:-1]
    ups = max(level - 1, 0)
    if ups > len(cur_parts):
        return module
    base_parts = cur_parts[: len(cur_parts) - ups]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts) if base_parts else module


def _candidate_modules(module_name: Optional[str]) -> List[str]:
    if not module_name:
        return []
    parts = module_name.split(".")
    return [".".join(parts[: idx + 1]) for idx in range(len(parts))]


def _local_import_graph(root: Path, module_map: Dict[str, Path]) -> Dict[Path, Set[Path]]:
    graph: Dict[Path, Set[Path]] = defaultdict(set)
    for module_name, rel in module_map.items():
        src = root / rel
        try:
            tree = ast.parse(src.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target_mod = alias.name
                    if target_mod in module_map:
                        graph[rel].add(module_map[target_mod])
                        continue
                    for cand in _candidate_modules(target_mod):
                        if cand in module_map:
                            graph[rel].add(module_map[cand])
            elif isinstance(node, ast.ImportFrom):
                base_mod = _resolve_import(
                    cur_module=module_name,
                    module=node.module,
                    level=int(node.level or 0),
                )
                if base_mod and base_mod in module_map:
                    graph[rel].add(module_map[base_mod])
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    qualified = f"{base_mod}.{alias.name}" if base_mod else alias.name
                    if qualified in module_map:
                        graph[rel].add(module_map[qualified])
    return graph


def _reachable(start_files: Iterable[Path], graph: Dict[Path, Set[Path]]) -> Set[Path]:
    seen: Set[Path] = set()
    q: deque[Path] = deque()
    for s in start_files:
        if s in seen:
            continue
        seen.add(s)
        q.append(s)
    while q:
        cur = q.popleft()
        for nxt in graph.get(cur, set()):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return seen


def _classify(
    rel: Path,
    runtime_used: Set[Path],
    training_used: Set[Path],
) -> Classification:
    if rel in runtime_used:
        return Classification(
            category="USED_CURRENT_MODEL_RUNTIME",
            used="yes",
            reason="reachable from runtime inference entrypoint(s)",
        )
    if rel in training_used:
        return Classification(
            category="USED_CURRENT_MODEL_CALIBRATION",
            used="yes",
            reason="reachable from calibration/diagnostic entrypoint(s)",
        )
    top = rel.parts[0] if rel.parts else ""
    suffix = rel.suffix.lower()
    if top == "results":
        return Classification(
            category="RESULT_ARTIFACT",
            used="no",
            reason="generated checkpoint/metric artifact; load only when explicitly provided",
        )
    if top == "tests":
        return Classification(
            category="TEST_ONLY",
            used="no",
            reason="used by test runs, not runtime model path",
        )
    if suffix in {".md", ".txt", ".png", ".jpg", ".jpeg", ".svg", ".tex"}:
        return Classification(
            category="DOC_OR_ASSET",
            used="no",
            reason="documentation or static asset",
        )
    if suffix in {".yaml", ".yml", ".json"}:
        return Classification(
            category="CONFIG_OR_DATA",
            used="optional",
            reason="used only when explicitly passed to scripts",
        )
    if suffix == ".py":
        return Classification(
            category="CODE_NOT_IN_CURRENT_MODEL_PATH",
            used="no",
            reason="python module/script outside current runtime+calibration dependency graph",
        )
    return Classification(
        category="OTHER",
        used="no",
        reason="not part of current model runtime path",
    )


def _write_report(
    *,
    root: Path,
    rows: List[Tuple[Path, Classification]],
    out_path: Path,
    include_venv: bool,
    venv_count: int,
    runtime_roots: List[Path],
    training_roots: List[Path],
) -> None:
    category_counts = Counter(row[1].category for row in rows)
    lines: List[str] = []
    lines.append("# Current Model File Usage Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Repository root: `{root}`")
    lines.append(f"- Files audited in table: `{len(rows)}`")
    if include_venv:
        lines.append("- `verification_env` included in table: `yes`")
    else:
        lines.append("- `verification_env` included in table: `no`")
        lines.append(f"- `verification_env` file count (excluded from table): `{venv_count}`")
    lines.append("")
    lines.append("## Current Model Scope")
    lines.append("")
    lines.append("- Runtime entrypoint roots:")
    for rel in runtime_roots:
        lines.append(f"  - `{rel.as_posix()}`")
    lines.append("- Calibration/diagnostic entrypoint roots:")
    for rel in training_roots:
        lines.append(f"  - `{rel.as_posix()}`")
    lines.append("")
    lines.append("## Category Counts")
    lines.append("")
    lines.append("| category | count |")
    lines.append("|---|---:|")
    for category, count in sorted(category_counts.items()):
        lines.append(f"| `{category}` | {count} |")
    lines.append("")
    lines.append("## Per-File Classification")
    lines.append("")
    lines.append("| path | category | used_in_current_model | reason |")
    lines.append("|---|---|---|---|")
    for rel, cls in rows:
        lines.append(
            f"| `{rel.as_posix()}` | `{cls.category}` | `{cls.used}` | {cls.reason} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-file report for what is used in the current model path."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("CURRENT_MODEL_FILE_USAGE_REPORT.md"),
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--include-venv",
        action="store_true",
        help="Include verification_env files in the table output.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    files_all = sorted(_iter_files(root, include_venv=True))
    venv_count = sum(1 for rel in files_all if rel.parts and rel.parts[0] == "verification_env")
    files = (
        files_all
        if args.include_venv
        else [rel for rel in files_all if not (rel.parts and rel.parts[0] == "verification_env")]
    )

    py_files = [rel for rel in files if rel.suffix == ".py"]
    module_map = _build_module_map(py_files)
    graph = _local_import_graph(root, module_map)

    runtime_roots = [
        Path("llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py"),
        Path("llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py"),
        Path("llama3_neuroplastic/sca_sparse_mlp.py"),
        Path("llama3_neuroplastic/sca_sparse_config.py"),
    ]
    training_roots = [
        Path("llama3_neuroplastic/run_sca_recalibration_from_hybrid_baseline.py"),
        Path("llama3_neuroplastic/init_learned_basis_from_dense_mlp.py"),
        Path("llama3_neuroplastic/run_sca_root_issue_locator.py"),
        Path("llama3_neuroplastic/run_sca_diagnostic_wipe.py"),
        Path("llama3_neuroplastic/run_decoder_mirror_sca_calibration.py"),
    ]

    runtime_start = [rel for rel in runtime_roots if rel in set(py_files)]
    training_start = [rel for rel in training_roots if rel in set(py_files)]
    runtime_used = _reachable(runtime_start, graph)
    training_used = _reachable(training_start, graph)

    rows: List[Tuple[Path, Classification]] = []
    for rel in files:
        cls = _classify(rel, runtime_used=runtime_used, training_used=training_used)
        rows.append((rel, cls))

    rows.sort(key=lambda x: x[0].as_posix().lower())
    out_path = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    _write_report(
        root=root,
        rows=rows,
        out_path=out_path,
        include_venv=bool(args.include_venv),
        venv_count=venv_count,
        runtime_roots=runtime_roots,
        training_roots=training_roots,
    )
    print(
        {
            "output": str(out_path),
            "rows": len(rows),
            "venv_count_excluded": 0 if args.include_venv else int(venv_count),
        }
    )


if __name__ == "__main__":
    main()
