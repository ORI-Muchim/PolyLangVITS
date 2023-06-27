import requests
import sys
import os


url1 = 'https://github.com/ORI-Muchim/PolyLangVITS/releases/download/v1.0/D_0.pth'
url2 = 'https://github.com/ORI-Muchim/PolyLangVITS/releases/download/v1.0/G_0.pth'

print("Downloading Discriminator Model...")

response1 = requests.get(url1, allow_redirects=True)

print("Downloading Generator Model...")

response2 = requests.get(url2, allow_redirects=True)

directory = f'./models/{sys.argv[2]}'

if not os.path.exists(directory):
    os.makedirs(directory)

discriminator_model = os.path.join(directory, 'D_0.pth')
generator_model = os.path.join(directory, 'G_0.pth')

with open(discriminator_model, 'wb') as file:
    file.write(response1.content)

print("Saving Discriminator Model...")

with open(generator_model, 'wb') as file:
    file.write(response2.content)

print("Saving Generator Model...\n")