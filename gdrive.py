#!/usr/bin/env python

import typing as ty

from .modules import install as install_package

try:
  import gdown
except ImportError:
  install_package('gdown')

from .common import resolve_path
from .console import stderr, stdout

# TODO(has) add all datasets from signal-public to this dictionary
SIGNAL_DATASETS = {
  "activity_triplets_V1_02182022": "1BUd2sAUP04Jf1Qnin0uhMeAXaN3gCyTy",
  "maintainers": '1g9hEzRp-EMOqr5AcoDJ8pdUjiwuWPn_2',
  "linux-kernel-data": "1h1AGfQkOhvgtcCR8tWVzObBSpBTURWM2",
  }


def download_data_from_google_drive(google_file_id: str, output_file_name: str, quiet_download: bool) -> str:
  try:
    file_path = resolve_path(f'./{output_file_name}')
    stdout.print(f"{output_file_name} already exists!")
  except ValueError:
    gdown.download(id=google_file_id, output=output_file_name, quiet=quiet_download)
    file_path = resolve_path(f'./{output_file_name}')

  return file_path


def get_dataset(dataset_name: str, is_local_file: bool, name2gdid: ty.Dict[str, str] = SIGNAL_DATASETS) -> str:
  file_path = f'./{dataset_name}'
  
  if is_local_file:
    file_path = resolve_path(file_path)
  else:
    if dataset_name in name2gdid:
      file_path = download_data_from_google_drive(
        name2gdid[dataset_name], 
        dataset_name, 
        quiet_download=True)
    else:
      stderr.print(f"Dataset {dataset_name} not found!")
      return None

  return file_path


if __name__ == "__main__":
  pass
