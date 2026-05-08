"""Smoke tests — each example compiles, launches, and (where applicable)
matches a torch reference."""

import importlib.util
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent


def _import(modname: str):
    spec = importlib.util.spec_from_file_location(modname, HERE / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def cuda_available():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")


def test_layouts(cuda_available):
    # Layout creation needs the DSL context; the example wraps it in @cute.jit.
    import cutlass.cute as cute
    mod = _import("00_layouts")
    cute.compile(mod.show_layouts)


def test_vector_add(cuda_available):
    _import("01_vector_add").run()


def test_tiled_copy(cuda_available):
    _import("02_tiled_copy_g2s").run()


def test_tiled_mma(cuda_available):
    _import("03_tiled_mma").run()


def test_flash_fwd_skeleton(cuda_available):
    # Skeleton — verifies compile + launch path only.
    _import("flash_fwd").run()
