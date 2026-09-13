"""Assemble an independent Windows package, excluding user data and all LLMs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    dist = ROOT / 'artifacts' / 'windows'
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onedir',
        '--name', 'Neurons', '--collect-all', 'ddgs', '--collect-all', 'primp', '--collect-all', 'lxml',
        '--distpath', str(dist), '--workpath', str(ROOT / '.runtime' / 'build'),
        '--specpath', str(ROOT / '.runtime'), str(ROOT / 'app.py')], cwd=ROOT, check=True)
    package = dist / 'Neurons'
    for filename in ('start.ps1', 'Run Neurons.cmd', 'README.md', 'NOTICE.md', 'original-data-verification.json'):
        shutil.copy2(ROOT / filename, package / filename)
    for folder in ('web', 'docs'):
        shutil.copytree(ROOT / folder, package / folder, dirs_exist_ok=True)
    target = package / '.data' / 'malecns'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / '.data' / 'malecns' / 'compiled', target / 'compiled', dirs_exist_ok=True)
    raw = target / 'raw'
    raw.mkdir(exist_ok=True)
    for source in (ROOT / '.data' / 'malecns' / 'raw').glob('*.feather'):
        shutil.copy2(source, raw / source.name)
    shutil.copy2(ROOT / '.data' / 'malecns' / 'download-manifest.json', target / 'download-manifest.json')
    # Include package metadata and license files for the embedded dependencies.
    site = Path(sys.executable).parent.parent / 'Lib' / 'site-packages'
    licenses = package / 'third-party'
    licenses.mkdir(exist_ok=True)
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    if python_license.exists():
        shutil.copy2(python_license, licenses / 'Python-LICENSE.txt')
    for metadata in site.glob('*.dist-info'):
        if metadata.name.lower().startswith(('numpy-', 'scipy-', 'pyinstaller-', 'pyinstaller_hooks_', 'altgraph-', 'pefile-', 'packaging-', 'ddgs-', 'primp-', 'lxml-', 'click-')):
            shutil.copytree(metadata, licenses / metadata.name, dirs_exist_ok=True)
    forbidden = [p for p in package.rglob('*') if p.is_file() and
                 (p.suffix in ('.gguf', '.sqlite3') or p.name.endswith(('.sqlite3-wal', '.sqlite3-shm')) or 'ollama' in p.parts)]
    if forbidden:
        raise RuntimeError('Unexpected user database or LLM files in package')
    archive = ROOT / 'artifacts' / 'Neurons-Windows-x64.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as target_zip:
        for item in sorted(package.rglob('*')):
            if item.is_file():
                target_zip.write(item, 'Neurons/' + item.relative_to(package).as_posix())
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    (ROOT / 'artifacts' / 'SHA256SUMS.txt').write_text(digest.hexdigest() + '  ' + archive.name + '\n', encoding='ascii')
    print(json.dumps({'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest.hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
