"""Install the official portable Ollama runtime inside this project (no system install)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'
URL = 'https://github.com/ollama/ollama/releases/download/v0.34.0/ollama-windows-amd64.zip'
SHA256 = 'a7dd1b174f39d3d1b8a25d4cbc86045d0e190b17187bfdcbe2f2ee3b5a11470e'


def main():
    RUNTIME.mkdir(exist_ok=True)
    archive = RUNTIME / 'ollama-windows-amd64.zip'
    partial = archive.with_suffix('.part')
    if not archive.exists():
        print('Downloading official Ollama v0.34.0 portable runtime (1.47 GB)', flush=True)
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as target:
            downloaded = 0
            previous = 0
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                downloaded += len(chunk)
                if downloaded - previous >= 128 * 1024 * 1024:
                    print(f'Downloaded {downloaded // (1024 * 1024)} MiB', flush=True)
                    previous = downloaded
        partial.replace(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError('Archive SHA-256 mismatch. Runtime was not extracted.')
    destination = (RUNTIME / 'ollama').resolve()
    destination.mkdir(exist_ok=True)
    print('SHA-256 verified. Extracting portable runtime...', flush=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            path = (destination / info.filename).resolve()
            if not path.is_relative_to(destination):
                raise RuntimeError('Unsafe archive path')
        bundle.extractall(destination)
    (RUNTIME / 'runtime-manifest.json').write_text(json.dumps({
        'url': URL, 'sha256': SHA256, 'version': 'v0.34.0',
        'model': 'qwen3.5:4b', 'modelSource': 'https://ollama.com/library/qwen3.5:4b'
    }, indent=2), encoding='utf-8')
    print('Portable runtime ready.', flush=True)


if __name__ == '__main__':
    main()
