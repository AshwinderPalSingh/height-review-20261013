"""Fetch checksum-pinned public packages into a new local cache for the Mac CPU recipe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.is_symlink(), 'Use a new package cache'
    rows = json.loads(Path(__file__).with_name('cpu-public-packages.json').read_text())
    names = set()
    for row in rows:
        name = row['filename']
        assert Path(name).name == name and name not in names
        names.add(name)
        url = urlparse(row['url'])
        assert url.scheme == 'https' and url.netloc == 'files.pythonhosted.org'
        assert Path(url.path).name == name
    args.output.mkdir(parents=True, exist_ok=False)

    def fetch(row):
        path = args.output / row['filename']
        subprocess.run(['curl', '--fail', '--silent', '--show-error', '--max-time', '300',
                        '--output', str(path), row['url']], check=True)
        assert path.stat().st_size == row['bytes'], path.name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path.name
        print('PACKAGE_VERIFIED=' + path.name, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, rows))


if __name__ == '__main__':
    main()
