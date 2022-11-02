import sys

from rich.console import Console
from rich.live import Live
from rich.progress import Progress

stdout = Console()
stderr = Console(file=sys.stderr)
progress_display = Progress(console=stderr)
live_display = Live(console=stderr, screen=False, auto_refresh=False)


if __name__ == "__main__":
  pass
