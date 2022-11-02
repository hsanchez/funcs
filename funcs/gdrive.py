import typing as ty

import gdown

from .colabs import resolve_path
from .console import stdout, stderr


def download_data_from_google_drive(google_file_id: str, output_file_name: str, quiet_download: bool) -> str:
  try:
    file_path = resolve_path(f'./{output_file_name}')
    stdout.print(f"{output_file_name} already exists!")
  except ValueError:
    gdown.download(id=google_file_id, output=output_file_name, quiet=quiet_download)
    file_path = resolve_path(f'./{output_file_name}')

  return file_path


def get_dataset(dataset_name: str, is_local_file: bool, name2gid: ty.Dict[str, str]) -> str:
  file_path = f'./{dataset_name}'
  
  if is_local_file:
    file_path = resolve_path(file_path)
  else:
    if dataset_name in name2gid:
      file_path = download_data_from_google_drive(
        name2gid[dataset_name], 
        dataset_name, 
        quiet_download=True)
    else:
      stderr.print(f"Dataset {dataset_name} not found!")
      return None

  return file_path


if __name__ == "__main__":
  pass
