import pytest

# TODO: long term, we might want this to be only the default, and for this parameter to be passed as a flag in testing
# TODO: fix visual tests to be easier to assess as correct or incorrect (title figures to give hint)
VISUAL_TEST = True

visualization = pytest.mark.skipif(not VISUAL_TEST, reason='No automatic tests for visualization tests')