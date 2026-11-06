import sys
from pathlib import Path

# make `import prep` work when running `pytest gpu_preprocessing/tests` from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
