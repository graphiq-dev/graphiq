# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Selecting the jax array library for the density-matrix backend.
"""

import importlib
import sys
import types

import numpy as np
import pytest

import graphiq
import graphiq.backends.density_matrix as dm_package


def _fake_jax(calls):
    # current jax: `jax.config` is the Config object, and there is no
    # `jax.config` submodule to import from
    jax = types.ModuleType("jax")
    jax.config = types.SimpleNamespace(update=lambda *args: calls.append(args))
    jax_numpy = types.ModuleType("jax.numpy")
    jax_numpy_linalg = types.ModuleType("jax.numpy.linalg")
    jax_numpy_linalg.eig = object()
    jax_scipy = types.ModuleType("jax.scipy")
    jax_scipy_linalg = types.ModuleType("jax.scipy.linalg")
    jax_scipy_linalg.eigh = object()
    jax.numpy, jax_numpy.linalg = jax_numpy, jax_numpy_linalg
    jax.scipy, jax_scipy.linalg = jax_scipy, jax_scipy_linalg
    return {
        "jax": jax,
        "jax.numpy": jax_numpy,
        "jax.numpy.linalg": jax_numpy_linalg,
        "jax.scipy": jax_scipy,
        "jax.scipy.linalg": jax_scipy_linalg,
    }


def test_jax_array_library_import_uses_the_jax_config_attribute(monkeypatch):
    calls = []
    modules = _fake_jax(calls)
    try:
        with monkeypatch.context() as m:
            for name, module in modules.items():
                m.setitem(sys.modules, name, module)
            m.setattr(graphiq, "DENSITY_MATRIX_ARRAY_LIBRARY", "jax")
            importlib.reload(dm_package)
            assert calls == [("jax_enable_x64", True)]
            assert dm_package.numpy is modules["jax.numpy"]
            assert dm_package.eig is modules["jax.numpy.linalg"].eig
            assert dm_package.eigh is modules["jax.scipy.linalg"].eigh
    finally:
        # the context has restored the flag and sys.modules; a reload keeps
        # old attributes, so drop the stand-in jax before rebinding numpy
        dm_package.__dict__.pop("jax", None)
        importlib.reload(dm_package)
    assert dm_package.numpy is np
    assert "jax" not in vars(dm_package)


def test_jax_array_library_imports_with_real_jax(monkeypatch):
    jax = pytest.importorskip("jax")
    try:
        with monkeypatch.context() as m:
            m.setattr(graphiq, "DENSITY_MATRIX_ARRAY_LIBRARY", "jax")
            importlib.reload(dm_package)
            assert dm_package.numpy is jax.numpy
            assert jax.config.jax_enable_x64
    finally:
        importlib.reload(dm_package)
    assert dm_package.numpy is np
