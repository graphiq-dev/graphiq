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
