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
from graphiq import DENSITY_MATRIX_ARRAY_LIBRARY

# todo, reload numpy/jax if needed? possible package conflicts
if DENSITY_MATRIX_ARRAY_LIBRARY == "jax":
    from jax.config import config

    config.update(
        "jax_enable_x64", True
    )  # 32-bit precision can lead to issues when comparing matrices

    import jax.numpy as numpy
    from jax.numpy.linalg import eig
    from jax.scipy.linalg import eigh
elif DENSITY_MATRIX_ARRAY_LIBRARY == "numpy":
    import numpy as numpy
    from numpy.linalg import eig
    from scipy.linalg import eigh
else:
    raise ImportError("Cannot load a valid array library. Options are 'numpy, 'jax'.")
