#!/usr/bin/env python

import subprocess
import sys


def install(package: str, quiet: bool = False, upgrade: bool = False):
  cmd = [sys.executable, "-m", "pip", "install"]
  if upgrade:
    cmd += ["--upgrade"]
  if quiet:
    cmd += ["-q"]
  
  cmd += [package]
  subprocess.check_call(cmd)


if __name__ == "__main__":
  pass