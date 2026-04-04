For Chatgpt/Codex:
- Use the fastest targeted command first
- Avoid repo-wide commands when a local command works
- Use `verification_env\Scripts\python` for repo-local Python checks that need installed packages like `torch`
- For 405B streaming runtime checks, prefer a short micro-probe first: keep the real CLI/runtime surface but cap with `--max-runtime-layers 2` or `4` and `--max-new-tokens 1` to isolate guard-layer vs sparse-layer cost before running the full 126-layer command
- For 405B decode validation, use `--max-new-tokens 2` as the smallest real end-to-end smoke test; `--max-new-tokens 1` only proves prefill + first sample and does not exercise the next-token decode pass
- For sparse-attention decode validation, `--attn-active-heads 64 --attn-max-active-heads 64` now means a 64-head candidate pool with budget-clamped live decode heads; on this 8 GB 2080 path the healthy log shape is `active_heads=16/128 ... min=16 max=64`, not live 64-head decode
- When `--attn-head-importance-path` is active, Taylor is intentionally auto-disabled for the streamed sparse-attention runtime unless `STREAMING_ALLOW_TAYLOR_WITH_SPARSE_ATTN=1` is set; use the default auto-disable for decode smoke tests unless you are explicitly debugging Taylor
- Update the AGENTS.md file if you discover a better way to organize the commands
