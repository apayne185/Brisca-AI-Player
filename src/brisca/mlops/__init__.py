"""Experiment tracking, model registry and tuning.

Requires the ``rl`` and ``mlops`` extras: ``pip install 'brisca[rl,mlops]'``.
"""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
