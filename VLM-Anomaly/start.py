import os
import urllib.request
import gdown
from zipfile import ZipFile
import re

# Change directory
#os.chdir('/content/MVFA-AD')

# Install requirements
os.system("pip install -r requirements.txt")

def download_file(url, filename):
    """Downloads a file from a given URL."""
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename} successfully.")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def download_and_unzip(id, output_path):
    """Downloads a zip file from a Google Drive link and extracts it."""
    try:
        # Download the zip file
        gdown.download(id, output_path, quiet=False)
        print(f"Downloaded to {output_path} successfully.")
        
        # Unzip the downloaded file
        with ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(output_path))
        print(f"Extracted {output_path} successfully.")
        
    except Exception as e:
        print(f"Error downloading or extracting: {e}")

# Ensure the directories exist
os.makedirs('./VLM/checkpoint', exist_ok=True)
os.makedirs('./checkpoint', exist_ok=True)

# Downloading files
url = "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
filename = "./VLM/checkpoint/ViT-L-14-336px.pt"
download_file(url, filename)

gdown.download("https://drive.google.com/uc?id=1ca3omEC0XGq5m-5qk2AFtFWGfj_F-HgD", "./resnet50_brain_regions.pth", quiet=False)
print(f"Downloaded Labelling model successfully.")

drive_link = "https://drive.google.com/uc?id=1bV1yzPxJarTRfd8liMIwyHcGywTTEL2k"
output_path = "./checkpoint/few-shot.zip"
download_and_unzip(drive_link, output_path)

# Go back to base directory
#os.chdir('/content/MVFA-AD')

# Download and extract additional data files
brain = "https://drive.google.com/uc?id=1YxcjcQqsPdkDO0rqIVHR5IJbqS9EIyoK"


os.makedirs('./data', exist_ok=True)

# Download and extract tar.gz files
tar_path = f"./data/Brain.tar.gz"
download_and_unzip(brain, tar_path)
os.system(f"tar -xvf {tar_path} -C ./data/")
