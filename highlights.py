#!/usr/bin/env python

import numpy as np

# style cells in pandas tables
def style_negative(v, props=''):
  return props if v < 0 else None

def highlight_eigenvalues(x, color):
  return np.where(x > 1.0, f"background-color: {color};", None)


def yellow_values(x):
  return np.where(x >= 0.7, f"background-color: yellow;", None)

def green_values(x):
  return np.where(((x >= 0.3) & (x < 0.7)), f"background-color: #e6ffe6;", None)

def pink_values(x):
  return np.where(((x >= 0.1) & (x < 0.3)), f"background-color: #ffe6e6;", None)


if __name__ == "__main__":
  pass
