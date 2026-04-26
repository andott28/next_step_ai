import re
import os

# 1. layer_selection.py
file_ls = r'c:\Users\andre\Desktop\Overføre\fullsparsepivot405b\llama3_neuroplastic\layer_selection.py'
with open(file_ls, 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace(
    '    if stripped == "" or stripped.lower() == "all":\n        return None if all_as_none else list(range(int(total_layers or 0)))',
    '    if stripped == "" or stripped.lower() == "all":\n        if not all_as_none and total_layers is None:\n            raise ValueError("Ambiguous layer selection: \'all\' requested but total_layers is not provided.")\n        return None if all_as_none else list(range(int(total_layers or 0)))'
)
code = code.replace(
    '    if total_layers is not None:\n        return sorted(idx for idx in selected if 0 <= idx < int(total_layers))\n    return sorted(selected)',
    '    if total_layers is not None:\n        for idx in selected:\n            if idx < 0 or idx >= int(total_layers):\n                raise ValueError(f"Layer index out of bounds: {idx}. Valid range is 0 to {int(total_layers) - 1}.")\n\n    return sorted(selected)'
)
with open(file_ls, 'w', encoding='utf-8') as f:
    f.write(code)

# 2. basis_fitting.py
file_bf = r'c:\Users\andre\Desktop\Overføre\fullsparsepivot405b\llama3_neuroplastic\basis_fitting.py'
with open(file_bf, 'r', encoding='utf-8') as f:
    code = f.read()

code = re.sub(
    r'(    decoder_bias = y_mean\.view\(num_blocks, int\(block_size\)\)\.contiguous\(\)\r?\n\r?\n    return \{\r?\n        )"encoder_weight":',
    r'\1"schema_version": 1,\n        "encoder_weight":',
    code
)
code = re.sub(
    r'(    decoder_bias = y_mean\.view\(num_blocks, int\(block_size\)\)\.contiguous\(\)\r?\n\r?\n)',
    r'\1    if decoder_blocks.shape != (num_blocks, int(basis_rank), int(block_size)):\n        raise RuntimeError(f"Shape mismatch: decoder_blocks shape {decoder_blocks.shape} != {(num_blocks, int(basis_rank), int(block_size))}")\n\n',
    code
)

code = re.sub(
    r'(    block_importance = scores_cpu\.mean\(dim=0\)\.contiguous\(\)\r?\n\r?\n    return \{\r?\n        )"encoder_weight":',
    r'\1"schema_version": 1,\n        "encoder_weight":',
    code
)
code = re.sub(
    r'(    block_importance = scores_cpu\.mean\(dim=0\)\.contiguous\(\)\r?\n\r?\n)',
    r'\1    if score_weight.shape != (num_blocks, int(basis_rank)):\n        raise RuntimeError(f"Shape mismatch: score_weight shape {score_weight.shape} != {(num_blocks, int(basis_rank))}")\n\n',
    code
)
with open(file_bf, 'w', encoding='utf-8') as f:
    f.write(code)

# 3. lm_head.py
file_lm = r'c:\Users\andre\Desktop\Overføre\fullsparsepivot405b\llama3_neuroplastic\experiments\runtime\lm_head.py'
with open(file_lm, 'r', encoding='utf-8') as f:
    code = f.read()

code = re.sub(
    r'print\(\s*f"\[lm_head\] staying off dense GPU load; free=.*?",\s*flush=True,\s*\)',
    r'self._record_runtime_event("lm_head_decision", action="staying_off_dense_gpu", free_bytes=free_bytes, required_bytes=required_bytes)',
    code, flags=re.DOTALL
)
code = re.sub(
    r'print\(\s*"\[lm_head\] source weight had no NF4 metadata.*?,\s*flush=True,\s*\)',
    r'self._record_runtime_event("lm_head_decision", action="quantized_dense_to_nf4_on_cpu")',
    code, flags=re.DOTALL
)
code = re.sub(
    r'print\(\s*f"\[lm_head\] resident on GPU as NF4:.*?,\s*flush=True,\s*\)',
    r'self._record_runtime_event("lm_head_decision", action="resident_on_gpu_nf4", weight_gb=self._lm_head_quantized_weight_gb(self._lm_head_nf4_meta_gpu))',
    code, flags=re.DOTALL
)
code = re.sub(
    r'print\(\s*f"\[lm_head\] GPU NF4 materialization failed.*?,\s*flush=True,\s*\)',
    r'self._record_runtime_event("lm_head_decision", action="nf4_materialization_failed", error=str(exc))',
    code, flags=re.DOTALL
)
code = re.sub(
    r'print\(\s*f"\[lm_head\] staying on CPU; free=.*?",\s*flush=True,\s*\)',
    r'self._record_runtime_event("lm_head_decision", action="staying_on_cpu", free_bytes=free_bytes, required_bytes=required_bytes)',
    code, flags=re.DOTALL
)
code = re.sub(
    r'print\(\s*f"\[lm_head\] resident on GPU: "\\s*f"\{float\(required_bytes\) / \(1024 \*\* 3\):\.2f\} GiB "\\s*f"\(free \{float\(gpu_free_bytes\) / \(1024 \*\* 3\):\.2f\} / \{float\(gpu_total_bytes\) / \(1024 \*\* 3\):\.2f\} GiB\)",\s*flush=True,\s*\)',
    r'self._record_runtime_event("lm_head_decision", action="resident_on_gpu", required_bytes=required_bytes, free_bytes=gpu_free_bytes, total_bytes=gpu_total_bytes)',
    code, flags=re.DOTALL
)
code = re.sub(
    r'print\(\s*f"\[lm_head\] resident on GPU: \{float\(required_bytes\).*?,\s*flush=True,\s*\)',
    r'self._record_runtime_event("lm_head_decision", action="resident_on_gpu", required_bytes=required_bytes)',
    code, flags=re.DOTALL
)
code = re.sub(
    r'print\(\s*f"\[lm_head\] GPU materialization failed; keeping CPU path.*?,\s*flush=True,\s*\)',
    r'self._record_runtime_event("lm_head_decision", action="materialization_failed", error=str(exc))',
    code, flags=re.DOTALL
)
code = re.sub(
    r'if rows != int\(active_idx\.shape\[0\]\):\s*active_idx = active_idx\.expand\(rows, -1\)\.contiguous\(\)',
    r'if getattr(self, "_lm_head_active_idx_cache", None) is None or self._lm_head_active_idx_cache.shape[0] < rows:\n                self._lm_head_active_idx_cache = active_idx.expand(rows, -1).contiguous()\n            active_idx = self._lm_head_active_idx_cache[:rows]',
    code
)
plan_func = """
    def get_execution_plan(self) -> dict[str, Any]:
        return {"lm_head": self.get_lm_head_status()}
"""
if "def get_execution_plan" not in code:
    code = code.replace("def get_lm_head_status(self) -> dict[str, Any]:", plan_func.lstrip() + "\n    def get_lm_head_status(self) -> dict[str, Any]:")
with open(file_lm, 'w', encoding='utf-8') as f:
    f.write(code)

# 4. token_posting_archive.py
file_tpa = r'c:\Users\andre\Desktop\Overføre\fullsparsepivot405b\llama3_neuroplastic\token_posting_archive.py'
with open(file_tpa, 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace(
    """        self._post_tok: dict[int, list[list[list[int]]]] = {}
        self._post_gen: dict[int, list[list[list[int]]]] = {}
        self._post_coeff: dict[int, list[list[list[int]]]] = {}
        self._post_scale: dict[int, list[list[list[float]]]] = {}""",
    """        self._post_tok: dict[int, np.ndarray] = {}
        self._post_gen: dict[int, np.ndarray] = {}
        self._post_coeff: dict[int, np.ndarray] = {}
        self._post_scale: dict[int, np.ndarray] = {}
        self._post_head: dict[int, np.ndarray] = {}
        self._post_count: dict[int, np.ndarray] = {}"""
)

code = code.replace(
    """            self._post_tok[layer_idx] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_gen[layer_idx] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_coeff[layer_idx] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_scale[layer_idx] = [[[] for _ in range(R)] for _ in range(G)]""",
    """            self._post_tok[layer_idx] = np.zeros((G, R, AC), dtype=np.int32)
            self._post_gen[layer_idx] = np.zeros((G, R, AC), dtype=np.int64)
            self._post_coeff[layer_idx] = np.zeros((G, R, AC), dtype=np.int8)
            self._post_scale[layer_idx] = np.zeros((G, R, AC), dtype=np.float32)
            self._post_head[layer_idx] = np.zeros((G, R), dtype=np.int32)
            self._post_count[layer_idx] = np.zeros((G, R), dtype=np.int32)"""
)

code = code.replace(
    """            self._post_tok[layer_idx] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_gen[layer_idx] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_coeff[layer_idx] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_scale[layer_idx] = [[[] for _ in range(R)] for _ in range(G)]""",
    """            self._post_tok[layer_idx].fill(0)
            self._post_gen[layer_idx].fill(0)
            self._post_coeff[layer_idx].fill(0)
            self._post_scale[layer_idx].fill(0)
            self._post_head[layer_idx].fill(0)
            self._post_count[layer_idx].fill(0)"""
)

code = code.replace(
    """                    self._post_tok[layer_idx][g][r].append(write_pos)
                    self._post_gen[layer_idx][g][r].append(int(n))
                    self._post_coeff[layer_idx][g][r].append(coeff)
                    self._post_scale[layer_idx][g][r].append(max_abs)""",
    """                    head = self._post_head[layer_idx][g, r]
                    self._post_tok[layer_idx][g, r, head] = write_pos
                    self._post_gen[layer_idx][g, r, head] = n
                    self._post_coeff[layer_idx][g, r, head] = coeff
                    self._post_scale[layer_idx][g, r, head] = max_abs
                    self._post_head[layer_idx][g, r] = (head + 1) % self.archive_capacity
                    self._post_count[layer_idx][g, r] += 1"""
)

probe_code_old = """        for r in top_r.tolist():
            beta_r = float(weighted[r])
            tok_list = self._post_tok[layer_idx][group_idx][r]
            gen_list = self._post_gen[layer_idx][group_idx][r]
            coeff_list = self._post_coeff[layer_idx][group_idx][r]
            scale_list = self._post_scale[layer_idx][group_idx][r]
            generation = self.archive_generation[layer_idx]
            for i in range(len(tok_list)):
                t = tok_list[i]
                if int(generation[t]) != int(gen_list[i]):
                    continue
                if stamp_buf[t] != step:
                    stamp_buf[t] = step
                    score_buf[t] = 0.0
                    touched.append(t)
                score_buf[t] += beta_r * (float(coeff_list[i]) / 127.0) * float(scale_list[i])"""

probe_code_new = """        for r in top_r.tolist():
            beta_r = float(weighted[r])
            limit = min(self._post_count[layer_idx][group_idx, r], self.archive_capacity)
            if limit == 0:
                continue
                
            tok_arr = self._post_tok[layer_idx][group_idx, r, :limit]
            gen_arr = self._post_gen[layer_idx][group_idx, r, :limit]
            coeff_arr = self._post_coeff[layer_idx][group_idx, r, :limit]
            scale_arr = self._post_scale[layer_idx][group_idx, r, :limit]
            
            valid_mask = (self.archive_generation[layer_idx][tok_arr] == gen_arr)
            
            valid_toks = tok_arr[valid_mask]
            valid_coeffs = coeff_arr[valid_mask]
            valid_scales = scale_arr[valid_mask]
            
            new_mask = stamp_buf[valid_toks] != step
            if new_mask.any():
                new_toks = valid_toks[new_mask]
                stamp_buf[new_toks] = step
                score_buf[new_toks] = 0.0
                touched.extend(new_toks.tolist())
                
            score_buf[valid_toks] += beta_r * (valid_coeffs.astype(np.float32) / 127.0) * valid_scales"""

code = code.replace(probe_code_old, probe_code_new)

with open(file_tpa, 'w', encoding='utf-8') as f:
    f.write(code)
