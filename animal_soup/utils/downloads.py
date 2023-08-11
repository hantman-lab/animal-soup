from pathlib import Path
import os
from tqdm import tqdm
from zipfile import ZipFile
import requests

sample_data_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.joinpath("tests", "sample_data")
pretrained_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.joinpath("pretrained_models")

os.makedirs(sample_data_dir, exist_ok=True)
os.makedirs(pretrained_dir, exist_ok=True)

sample_data_file = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.joinpath("tests", "sample_data.zip")
pretrained_file = Path(os.path.dirname(os.path.abspath(__file__))).parent.joinpath("pretrained_models.zip")


def download_data():
    """Downloads sample data from Zenodo."""

    if len(list(sample_data_dir.iterdir())) == 0:

        print(f"Downloading sample data")
        url = 'https://zenodo.org/record/8225385/files/sample_data.zip'

        # basically from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        with open(sample_data_file, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        ZipFile(sample_data_file).extractall(sample_data_dir.parent)
    else:
        print("Sample data already downloaded")


def download_models():
    """Downloads pre-trained models from Zenodo"""

    if len(list(pretrained_dir.iterdir())) == 0:
        print(f"Downloading pre-trained models")
        url = 'https://zenodo.org/record/8225385/files/pretrained_models.zip'

        # basically from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        with open(pretrained_file, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        ZipFile(pretrained_file).extractall(pretrained_dir.parent)
    else:
        print("Pre-trained models already downloaded")


if __name__ == "__main__":
    print(sample_data_dir)
    print(pretrained_dir)
    print(sample_data_file)
    print(pretrained_file)
