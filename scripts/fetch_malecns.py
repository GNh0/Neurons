"""Download official MaleCNS v1.0 tables, checking GCS MD5 and recording provenance."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {
    'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'neurotransmitters': 'body-neurotransmitters-male-cns-v1.0.feather',
    'connections': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
}


def download(kind, destination):
    filename = FILES[kind]
    path = destination / filename
    url = BASE + filename
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
        expected_size = int(response.headers['Content-Length'])
        hashes = response.headers.get('x-goog-hash', '')
        expected_md5 = next((x.strip().split('=', 1)[1] for x in hashes.split(',')
                             if x.strip().startswith('md5=')), None)
    if not path.exists() or path.stat().st_size != expected_size:
        partial = path.with_suffix('.partial')
        print(f'Downloading {kind}: {expected_size / 1e6:.1f} MB', flush=True)
        with urllib.request.urlopen(url, timeout=90) as response, partial.open('wb') as target:
            received, last_report = 0, time.monotonic()
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                received += len(chunk)
                if time.monotonic() - last_report > 10:
                    print(f'{kind}: {received / expected_size:.0%}', flush=True)
                    last_report = time.monotonic()
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f'Incomplete download: {filename}')
        check_path = partial
    else:
        check_path = path
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with check_path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            md5.update(chunk)
            sha256.update(chunk)
    observed_md5 = base64.b64encode(md5.digest()).decode()
    if expected_md5 and observed_md5 != expected_md5:
        raise RuntimeError(f'MD5 mismatch: {filename}')
    if check_path != path:
        check_path.replace(path)
    print(f'Verified {kind}: {path.stat().st_size:,} bytes', flush=True)
    return {'file': filename, 'url': url, 'bytes': expected_size, 'sha256': sha256.hexdigest(),
            'gcs_md5': expected_md5, 'md5_verified': expected_md5 == observed_md5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', choices=list(FILES))
    args = parser.parse_args()
    directory = ROOT / '.data' / 'malecns' / 'raw'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory.parent / 'download-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {
        'dataset': 'MaleCNS v1.0', 'source': 'https://male-cns.janelia.org/download/',
        'license': 'CC-BY-4.0', 'publication': 'https://doi.org/10.1016/j.cell.2026.08.015',
        'data_release': '2026-06-08', 'paper_publication': '2026-09-03', 'files': {},
    }
    for kind in args.only or FILES:
        manifest['files'][kind] = download(kind, directory)
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(manifest_path)


if __name__ == '__main__':
    main()
