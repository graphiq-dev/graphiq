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
