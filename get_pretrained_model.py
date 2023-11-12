import requests
import sys
import os

def download_file(url, path):
    print(f"Downloading {os.path.basename(path)}...")
    response = requests.get(url, allow_redirects=True)
    with open(path, 'wb') as file:
        file.write(response.content)
    print(f"Saved {os.path.basename(path)}.")

def get_model(model_type):
    gen = f"./models/{model_type}/G_0.pth"
    if not os.path.isfile(gen):
        model_urls = {
            'D_0.pth': 'https://github.com/ORI-Muchim/PolyLangVITS/releases/download/v1.0/D_0.pth',
            'G_0.pth': 'https://github.com/ORI-Muchim/PolyLangVITS/releases/download/v1.0/G_0.pth'
        }

        directory = f'./models/{model_type}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        for filename, url in model_urls.items():
            file_path = os.path.join(directory, filename)
            download_file(url, file_path)
    else:
        print('Skipping Download... Model exists.')

if __name__ == "__main__":
    model_type = sys.argv[2]
    get_model(model_type)
