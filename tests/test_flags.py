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
import pytest

import graphiq

# TODO: long term, we might want this to be only the default, and for this parameter to be passed as a flag in testing
# TODO: fix visual tests to be easier to assess as correct or incorrect (title figures to give hint)

VISUAL_TEST = True
if graphiq.DENSITY_MATRIX_ARRAY_LIBRARY == "jax":
    try:
        import jax

        JAX_TEST = True
    except:
        JAX_TEST = False
else:
    JAX_TEST = False

visualization = pytest.mark.skipif(
    not VISUAL_TEST, reason="No automatic tests for visualization tests"
)

jax_library = pytest.mark.skipif(not JAX_TEST, reason="JAX is not installed")
