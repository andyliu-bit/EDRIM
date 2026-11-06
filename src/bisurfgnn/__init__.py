"""BiSurfGNN research package."""

from .config import ExperimentConfig, load_config
from .models.bisurf import BiSurfGNN

__all__ = ["BiSurfGNN", "ExperimentConfig", "load_config"]
__version__ = "0.1.0"

