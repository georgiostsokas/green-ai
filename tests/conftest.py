import os

# Tests always use the reference data bundled with the package, never a user cache.
os.environ["GREEN_AI_USE_BUNDLED"] = "1"
