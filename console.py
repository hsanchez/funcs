#!/usr/bin/env python

import sys

from .pinstall import install as install_package

try:
  from rich.console import Console
  from rich.live import Live
  from rich.progress import Progress
except ImportError:
  install_package('rich')
  from rich.console import Console
  from rich.live import Live
  from rich.progress import Progress


stdout = Console()
stderr = Console(file=sys.stderr)
progress_display = Progress(console=stderr)
live_display = Live(console=stderr, screen=False, auto_refresh=False)


if __name__ == "__main__":
  pass
