#!/usr/bin/env python

import subprocess
import sys
import typing as ty


def install(package: str, quiet: bool = False, upgrade: bool = False):
  cmd = [sys.executable, "-m", "pip", "install"]
  if upgrade:
    cmd += ["--upgrade"]
  if quiet:
    cmd += ["-q"]
  
  cmd += [package]
  subprocess.check_call(cmd)


def import_module(module: str, base_package: str = None) -> ty.Any:
  try:
    import importlib
  except ImportError:
    install('importlib')
    import importlib
    
  try:
    return importlib.import_module(module)
  except ImportError:
    if base_package is None:
      install(module)
    else:
      install(base_package)
    return importlib.import_module(module)


if __name__ == "__main__":
  pass