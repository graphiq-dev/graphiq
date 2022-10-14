from src import DENSITY_MATRIX_ARRAY_LIBRARY

# todo, reload numpy/jax if needed? possible package conflicts
if DENSITY_MATRIX_ARRAY_LIBRARY == "jax":
    import jax.numpy as numpy
    from jax.scipy.linalg import eigh
elif DENSITY_MATRIX_ARRAY_LIBRARY == "numpy":
    import numpy as numpy
    from scipy.linalg import eigh
else:
    raise ImportError("Cannot load a valid array library. Options are 'numpy, 'jax'.")
