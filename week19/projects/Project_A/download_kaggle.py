import sys
import pathlib
import shutil

try:
    import kagglehub
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])
    import kagglehub

DATASET = "mlg-ulb/creditcardfraud"
out_dir = pathlib.Path('data')
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Downloading {DATASET} ...")
path = kagglehub.dataset_download(DATASET)
print("Path to dataset files:", path)

src = pathlib.Path(path)
for item in src.iterdir():
    dest = out_dir / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)
print(f"Done. Files copied to {out_dir.resolve()}")
