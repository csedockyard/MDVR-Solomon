import urllib.request
from pathlib import Path

def download_solomon_instance(instance_name="c101"):
    # 1. Get the absolute path of the directory this script is inside (the 'src' folder)
    script_dir = Path(__file__).parent
    
    # 2. Go up one level to 'MLpro', then into the 'data' folder
    data_dir = script_dir.parent / "data"
    
    # 3. Create the folder if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    file_path = data_dir / f"{instance_name.lower()}.txt"

    url = f"https://raw.githubusercontent.com/jonzhaocn/VRPTW-ACO-python/master/solomon-100/{instance_name.lower()}.txt"
    
    print(f"Initiating secure download of {instance_name.upper()}...")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Success! Saved precisely to: {file_path}")
    except Exception as e:
        print(f"Download failed. Error: {e}")

if __name__ == "__main__":
    download_solomon_instance("c101")