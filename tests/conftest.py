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

# Force a non-interactive matplotlib backend for the whole test suite, so that
# tests exercising draw_circuit()/plt.show() etc. don't pop up plot windows.
# This must run before matplotlib.pyplot is imported anywhere else, which is
# why it lives in the top-level conftest.py: pytest loads it before collecting
# any test module (and before their own conftest.py files import graphiq).
import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _mixture_fold_flag_is_restored():
    """
    Fail the test that leaves ``REDUCE_STABILIZER_MIXTURE`` flipped.

    The flag is a module global read on every depolarizing noise application; a test that
    sets it False without restoring it turns folding off for every later test in the
    process, and noisy solver runs then grow their mixtures without bound. Autouse so this
    is set up before ``monkeypatch`` and torn down after it.
    """
    yield
    import graphiq.noise.noise_models as nm

    assert nm.REDUCE_STABILIZER_MIXTURE is True, (
        "this test left graphiq.noise.noise_models.REDUCE_STABILIZER_MIXTURE = "
        f"{nm.REDUCE_STABILIZER_MIXTURE!r}; restore it with monkeypatch.setattr"
    )
