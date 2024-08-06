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

drive_link = "https://drive.google.com/uc?id=1bV1yzPxJarTRfd8liMIwyHcGywTTEL2k"
output_path = "./checkpoint/few-shot.zip"
download_and_unzip(drive_link, output_path)

drive_link = "https://drive.google.com/uc?id=1nGhcK32CrkgTR5Rav6rNfptUHaASfRnU"
output_path = "./checkpoint/zero-shot.zip"
download_and_unzip(drive_link, output_path)

# Unzip files
os.system("unzip ./checkpoint/few-shot.zip -d ./checkpoint")
os.system("unzip ./checkpoint/zero-shot.zip -d ./checkpoint")

# Go back to base directory
#os.chdir('/content/MVFA-AD')

# Download and extract additional data files
s = """Liver: https://drive.google.com/file/d/1xriF0uiwrgoPh01N6GlzE5zPi_OIJG1I/view?usp=sharing
Brain: https://drive.google.com/file/d/1YxcjcQqsPdkDO0rqIVHR5IJbqS9EIyoK/view?usp=sharing
HIS: https://drive.google.com/file/d/1hueVJZCFIZFHBLHFlv1OhqF8SFjUVHk6/view?usp=sharing
RESC: https://drive.google.com/file/d/1BqDbK-7OP5fUha5zvS2XIQl-_t8jhTpX/view?usp=sharing
OCT17: https://drive.google.com/file/d/1GqT0V3_3ivXPAuTn4WbMM6B9i0JQcSnM/view?usp=sharing
ChestXray: https://drive.google.com/file/d/15DhnBAk-h6TGLTUbNLxP8jCCDtwOHAjb/view?usp=sharing"""

pattern = r'/d/(.*?)/view'
ids = re.findall(pattern, s)
print(ids)

pattern = r'(.*?): '
names = re.findall(pattern, s)
print(names)

# Download and extract tar.gz files
for id, name in zip(ids, names):
    tar_path = f"./data/{name}.tar.gz"
    download_and_unzip(f"https://drive.google.com/uc?id={id}", tar_path)
    os.system(f"tar -xvf {tar_path} -C ./data/")