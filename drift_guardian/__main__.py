"""``python -m drift_guardian`` — то же, что консольная команда ``drift-guardian``."""
import sys

from .cli import main

sys.exit(main())
