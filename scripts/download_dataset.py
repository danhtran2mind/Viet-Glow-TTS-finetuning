import os
import urllib.request
import tarfile
import argparse

def download_dataset(url, file_name):
    """Download the dataset from the specified URL and save it as file_name."""
    urllib.request.urlretrieve(url, file_name)
    print(f"Downloaded dataset to {file_name}")

def extract_dataset(file_name, output_dir):
    """Extract the tar.gz file to the specified output directory and remove the tar.gz file."""
    os.makedirs(output_dir, exist_ok=True)
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=output_dir)
    os.remove(file_name)
    print(f"Extracted dataset to {output_dir} and removed {file_name}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download and extract a dataset from a URL.")
    parser.add_argument('--url', type=str, 
                        default="https://huggingface.co/datasets/ntt123/viet-tts-dataset/resolve/main/viet-tts.tar.gz",
                        help="URL of the dataset to download (default: viet-tts dataset)")
    parser.add_argument('--output-dir', type=str, default="data",
                        help="Directory to extract the dataset (default: data)")
    parser.add_argument('--file-name', type=str, default="viet-tts.tar.gz",
                        help="Name of the downloaded tar.gz file (default: viet-tts.tar.gz)")

    # Parse arguments
    args = parser.parse_args()

    # Execute download and extraction
    download_dataset(args.url, args.file_name)
    extract_dataset(args.file_name, args.output_dir)

if __name__ == "__main__":
    main()