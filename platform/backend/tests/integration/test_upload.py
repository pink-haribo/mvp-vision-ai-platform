"""Test script for dataset image upload API."""
import requests
import time
from pathlib import Path

# Read token and dataset ID
with open('test_token.txt', 'r') as f:
    token = f.read().strip()

dataset_id = '5af56211-c1a3-4c3a-a2ba-2a732b342e2f'

headers = {
    'Authorization': f'Bearer {token}'
}

# Prepare files with webkitRelativePath in filename
files = []
test_dir = Path('test_images')

for img_path in test_dir.rglob('*.jpg'):
    # Get relative path from test_images (include folder name)
    relative_path = img_path.relative_to(test_dir.parent)

    # Open file and add to files list
    # filename must include the folder structure (use forward slash for web compatibility)
    relative_path_str = str(relative_path).replace('\\', '/')
    files.append((
        'files',
        (relative_path_str, open(img_path, 'rb'), 'image/jpeg')
    ))

print(f'Uploading {len(files)} images...')
print(f'Files:')
for f in files:
    print(f'  - {f[1][0]}')

# Upload with timing
start_time = time.time()

response = requests.post(
    f'http://localhost:8000/api/v1/datasets/{dataset_id}/upload-images',
    files=files,
    headers=headers
)

elapsed = time.time() - start_time

# Close files
for _, (_, fp, _) in files:
    fp.close()

print(f'\n[TIMING] Upload completed in {elapsed:.2f} seconds')
print(f'Status: {response.status_code}')

if response.status_code == 200:
    data = response.json()
    print(f'\n[SUCCESS] {data["status"].upper()}')
    print(f'Message: {data["message"]}')
    if 'metadata' in data:
        meta = data['metadata']
        print(f'\nDataset Info:')
        print(f'  - Name: {meta["dataset_name"]}')
        print(f'  - Total images: {meta["num_images"]}')
        print(f'  - Folder structure: {meta.get("folder_structure", {})}')
        print(f'  - Storage path: {meta.get("storage_path", "")}')
else:
    print(f'[ERROR] {response.text}')
