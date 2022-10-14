DENSITY_MATRIX_ARRAY_LIBRARY = "jax"

if DENSITY_MATRIX_ARRAY_LIBRARY == "jax":
    import jax.numpy as numpy
elif DENSITY_MATRIX_ARRAY_LIBRARY == "numpy":
    import numpy as numpy
else:
    raise ImportError("Cannot load a valid array library. Options are 'numpy, 'jax'.")
