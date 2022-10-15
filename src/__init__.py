DENSITY_MATRIX_ARRAY_LIBRARY = "jax"

#%%
import jax.numpy as np
# from jax.numpy.linalg import
a = 2 * np.eye(2)

b = np.eye(2)

n = np.linalg.norm(a - b, ord="fro")

print(n)
