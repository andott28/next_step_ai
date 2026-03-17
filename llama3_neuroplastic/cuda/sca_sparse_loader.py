from __future__ import annotations

import shutil
import sysconfig
import tempfile
from pathlib import Path
import threading
import os
from typing import Tuple

import torch
from torch.utils.cpp_extension import load


_EXT_LOCK = threading.Lock()
_EXT_MODULE = None
_EXT_ERROR = None


def _stage_sources_in_ascii_temp(src_dir: Path) -> list[str]:
    """
    Ninja/MSVC can fail on non-ASCII source paths on Windows.
    Stage sources into a deterministic ASCII temp directory.
    """
    stage_dir = Path(tempfile.gettempdir()) / "sca_cuda_ascii_src"
    stage_dir.mkdir(parents=True, exist_ok=True)
    filenames = ["sca_sparse_bindings.cpp", "sca_sparse_kernels.cu"]
    staged: list[str] = []
    for name in filenames:
        src = src_dir / name
        dst = stage_dir / name
        shutil.copy2(src, dst)
        staged.append(str(dst))
    return staged


def _windows_short_path(path: Path) -> str:
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        res = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))
        if res > 0:
            return str(buf.value)
    except Exception:
        pass
    return str(path)


def _extra_ascii_include_paths() -> list[str]:
    torch_root = Path(torch.__file__).resolve().parent
    include_dirs = [
        torch_root / "include",
        torch_root / "include" / "torch" / "csrc" / "api" / "include",
    ]
    py_inc = sysconfig.get_paths().get("include")
    if py_inc:
        include_dirs.append(Path(py_inc))
    out: list[str] = []
    for inc in include_dirs:
        if inc.exists():
            out.append(_windows_short_path(inc))
    return out


def _ensure_msvc_cl_on_path() -> None:
    if shutil.which("cl") is not None:
        return
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        return
    try:
        import subprocess

        install_path = (
            subprocess.check_output(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return
    if not install_path:
        return
    msvc_root = Path(install_path) / "VC" / "Tools" / "MSVC"
    if not msvc_root.exists():
        return
    candidates = sorted(msvc_root.glob("*/bin/Hostx64/x64/cl.exe"), reverse=True)
    if not candidates:
        return
    cl_dir = str(candidates[0].parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    if cl_dir not in path_entries:
        os.environ["PATH"] = cl_dir + os.pathsep + os.environ.get("PATH", "")


class SCACUDAExtensionError(RuntimeError):
    pass


def load_sca_cuda_extension(verbose: bool = False):
    global _EXT_MODULE, _EXT_ERROR

    with _EXT_LOCK:
        if _EXT_MODULE is not None:
            return _EXT_MODULE
        if _EXT_ERROR is not None:
            raise SCACUDAExtensionError(str(_EXT_ERROR)) from _EXT_ERROR

        src_dir = Path(__file__).parent
        sources = _stage_sources_in_ascii_temp(src_dir)
        _ensure_msvc_cl_on_path()
        ninja = shutil.which("ninja")
        if ninja is None:
            py_scripts = Path(sysconfig.get_paths().get("scripts", ""))
            ninja_candidate = py_scripts / ("ninja.exe" if os.name == "nt" else "ninja")
            if ninja_candidate.exists():
                os.environ["PATH"] = str(py_scripts) + os.pathsep + os.environ.get("PATH", "")

        try:
            _EXT_MODULE = load(
                name="sca_sparse_cuda_ext",
                sources=sources,
                extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler"],
                extra_cflags=["-O3"],
                extra_include_paths=_extra_ascii_include_paths(),
                verbose=verbose,
            )
            return _EXT_MODULE
        except Exception as exc:  # pragma: no cover
            _EXT_ERROR = exc
            raise SCACUDAExtensionError(
                "Failed to build/load SCA CUDA extension. "
                "Set sca_use_cuda=False to run without CUDA kernels. "
                "If this is Windows, ensure MSVC Build Tools + Windows SDK + Ninja are in PATH."
            ) from exc


class SCACUDAKernels:
    def __init__(self, module) -> None:
        self._module = module

    @classmethod
    def from_build(cls, verbose: bool = False) -> "SCACUDAKernels":
        return cls(load_sca_cuda_extension(verbose=verbose))

    def spatial_gate(
        self,
        q: torch.Tensor,
        block_centers: torch.Tensor,
        refractory_until: torch.Tensor,
        step: int,
        sigma: float,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._module.spatial_gate_cuda(
            q,
            block_centers,
            refractory_until,
            int(step),
            float(sigma),
            int(top_k),
        )

    def sparse_adapter(
        self,
        hidden_flat: torch.Tensor,
        active_idx: torch.Tensor,
        down_w: torch.Tensor,
        down_b: torch.Tensor,
        up_w: torch.Tensor,
        up_b: torch.Tensor,
    ) -> torch.Tensor:
        return self._module.sparse_adapter_cuda(
            hidden_flat,
            active_idx,
            down_w,
            down_b,
            up_w,
            up_b,
        )
