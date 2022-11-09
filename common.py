#!/usr/bin/env python

import datetime
import functools
import itertools
import os
import pathlib
import shutil
import typing as ty
from itertools import islice
from timeit import default_timer as timer
from types import ModuleType

from .console import new_live_display, stderr

Decoratee = ty.TypeVar('Decoratee', bound=ty.Callable[..., ty.Any])
OutputType = ty.TypeVar("OutputType")
PathLike = ty.Union[str, pathlib.Path]


def set_default_vars(os_env: dict, extra_builtins: ty.Union[ModuleType, dict] = None, ipython_val: ty.Any = None) -> ty.Union[ModuleType, dict]:
  global __builtins__
  os.environ.update(os_env)
  try:
    if isinstance(__builtins__, dict):
      if isinstance(extra_builtins, dict):
        __builtins__.update(extra_builtins)
      else:
        __builtins__.update(extra_builtins.__dict__)
      if ipython_val is not None:
        is_colab = 'google.colab' in str(ipython_val)
        __builtins__.update({'__IS_COLAB__': is_colab})
    else:
      if isinstance(extra_builtins, dict):
        __builtins__.__dict__.update(extra_builtins)
      else:
        __builtins__.__dict__.update(extra_builtins.__dict__)
      if ipython_val is not None:
        is_colab = 'google.colab' in str(ipython_val)
        setattr(__builtins__, '__IS_COLAB__', is_colab)
  except Exception as e:
    stderr.print(f"Failed to update __builtins__ with {extra_builtins}")
    raise e
  
  return __builtins__


# thx to https://stackoverflow.com/questions/53581278
def is_run_in_colab(extra_builtins: ty.Any = None, get_ipython_fn: ty.Callable[..., ty.Any] = None) -> bool:
  if 'google.colab' in os.environ['PATH']:
    return True
  elif hasattr(__builtins__,'__IPYTHON__') or hasattr(extra_builtins,'__IPYTHON__'):
    if hasattr(__builtins__, '__IS_COLAB__'):
      return __builtins__.__IS_COLAB__
    else:
      from IPython import get_ipython
      return 'google.colab' in str(get_ipython()) or 'google.colab' in str(get_ipython_fn())
  return False


# a few helper methods (borrowed from our Colabs)
def take(n, iterable):
  "Return first n items of the iterable as a list"
  return list(islice(iterable, n))


def take_from_dict(n, d):
  return take(n, d.items())


def partition(items: ty.Iterable[OutputType], chunk_size: ty.Optional[int]) -> ty.Iterable[ty.List[OutputType]]:
  positive_int = isinstance(chunk_size, int) and chunk_size > 0
  if not(chunk_size is None or positive_int):
    raise ValueError("Chunk size must be a positive int (or None)")

  iterator = iter(items)
  part = list(itertools.islice(iterator, chunk_size))
  while part:
    yield part
    part = list(itertools.islice(iterator, chunk_size))


def resolve_path(file_to_resolve):  
  if file_to_resolve is None:
    raise ValueError(f"Failed to file to resolve. See {file_to_resolve}")

  # thx to click.Path for this trick
  # os.path.realpath doesn't resolve symlinks on Windows
  # until Python 3.8. Use pathlib for now.
  resolved = os.fsdecode(pathlib.Path(file_to_resolve).resolve())
  if os.path.exists(resolved):
    return resolved

  raise ValueError(f"Failed to resolve {resolved}")


def mkdir(newdir: ty.Any) -> None:
  if isinstance(newdir, str):
    if not os.path.isdir(newdir):
      os.makedirs(newdir)
  elif isinstance(newdir, pathlib.Path):
    if not newdir.is_dir():
      newdir.mkdir()


def is_dir_empty(dir_path: pathlib.Path) -> bool:
  # thx to https://stackoverflow.com/questions/25675352
  if not dir_path:
    return True
  return not any(dir_path.iterdir())


def is_file_real(f: PathLike) -> bool:
  try:
    return pathlib.Path(resolve_path(f)).exists()
  except Exception:
    return False


def with_status(console: ty.Any, prefix: str, suffix: ty.Callable[[ty.Any], str] = lambda _ : '') -> ty.Callable[[Decoratee], Decoratee]:
  """This decorator writes the prefix, followed by three dots, then runs the
  decorated function.  Upon success, it appends check mark, upon failure, it appends
  an cross mark.  If suffix is set, the result of the computation is passed to suffix,
  and the resulting string is appended after check mark."""
  def decorator(func: Decoratee) -> Decoratee:
    @functools.wraps(func)
    def wrapper(*args: ty.Any, **kwargs: ty.Any) -> ty.Any:
      with new_live_display(console) as live:
        live.update(f"{prefix} ...", refresh=True)
        start = timer()
        try:
          result = func(*args, **kwargs)
        except Exception as e:
          live.update(
            f"{prefix} [red][:heavy_multiplication_x:]",
            refresh=True,
          )
          raise e
        end = timer()
        elapsed = datetime.timedelta(seconds=end - start)
        live.update(
          f"{prefix} [green][:heavy_check_mark:] ({elapsed}) {suffix(result)}",
          refresh=True,
        )
      return result
    return ty.cast(Decoratee, wrapper)

  return decorator


def move(src, dst):
  if os.path.exists(src):
    shutil.move(src, dst)


def rotate_file(f):
  if isinstance(f, pathlib.Path):
    f = str(f)

  i = 1
  fileformat = "{}_{}"
  (filename, ext) = os.path.splitext(f)
  
  while os.path.exists(fileformat.format(filename, i) + ext):
    i += 1

  newfile = fileformat.format(filename, i) + ext
  move(f, newfile)


if __name__ == "__main__":
  pass
