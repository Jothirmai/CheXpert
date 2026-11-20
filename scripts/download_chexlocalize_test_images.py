"""Download the authorized CheXpert test images from a short-lived Azure SAS URL.

The SAS URL is read with no terminal echo and is never stored in files or logs.
Only CheXpert/test radiographs are downloaded.
"""

import base64
import concurrent.futures
import csv
import getpass
import hashlib
import json
import os
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


ROOT = Path('data/chexpert_test_500')
DEST = ROOT / 'images'
PREFIX = 'CheXpert/test/'
THREAD_LOCAL = threading.local()


def session():
    if not hasattr(THREAD_LOCAL, 'session'):
        THREAD_LOCAL.session = requests.Session()
    return THREAD_LOCAL.session


def list_blobs(sas):
    parsed = urllib.parse.urlsplit(sas)
    if parsed.scheme != 'https' or parsed.netloc != 'aimistanforddatasets01.blob.core.windows.net' or parsed.path != '/chexlocalize':
        raise RuntimeError('Unexpected Azure container URL')
    query = urllib.parse.parse_qs(parsed.query)
    if not {'sig', 'sp', 'se'} <= set(query) or 'r' not in query['sp'][0] or 'l' not in query['sp'][0]:
        raise RuntimeError('SAS URL lacks read/list access')
    query.update({'restype': ['container'], 'comp': ['list'], 'prefix': [PREFIX], 'maxresults': ['5000']})
    url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path,
                                   urllib.parse.urlencode(query, doseq=True), ''))
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    if root.findtext('NextMarker'):
        raise RuntimeError('More than 5000 test blobs; unexpected container layout')
    blobs = []
    for item in root.findall('.//Blob'):
        name = item.findtext('Name')
        size = int(item.findtext('Properties/Content-Length'))
        md5 = item.findtext('Properties/Content-MD5')
        if not name.startswith(PREFIX) or not name.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise RuntimeError('Unexpected test blob name')
        blobs.append((name, size, md5))
    if len(blobs) != 668:
        raise RuntimeError(f'Expected 668 test radiographs; found {len(blobs)}')
    studies = {str(Path(name).parent) for name, _, _ in blobs}
    if len(studies) != 500:
        raise RuntimeError(f'Expected 500 test studies; found {len(studies)}')
    with (ROOT / 'groundtruth.csv').open(newline='') as handle:
        reference = {row['Study'].replace('CheXpert-v1.0/', 'CheXpert/')
                     for row in csv.DictReader(handle)}
    if studies != reference:
        raise RuntimeError('Test image studies do not match frozen expert labels')
    return parsed, blobs


def fetch(parsed, blob):
    name, expected_size, expected_md5 = blob
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_size:
        return name, expected_size, hashlib.sha256(target.read_bytes()).hexdigest(), True
    blob_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc,
                                        parsed.path + '/' + urllib.parse.quote(name, safe='/'),
                                        parsed.query, ''))
    for attempt in range(3):
        temporary = target.with_suffix(target.suffix + '.part')
        try:
            digest = hashlib.sha256()
            md5 = hashlib.md5()
            count = 0
            with session().get(blob_url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        digest.update(chunk)
                        md5.update(chunk)
                        count += len(chunk)
            if count != expected_size:
                raise RuntimeError(f'Incorrect size for {name}: {count} != {expected_size}')
            if expected_md5 and base64.b64encode(md5.digest()).decode() != expected_md5:
                raise RuntimeError(f'Incorrect MD5 for {name}')
            os.replace(temporary, target)
            return name, count, digest.hexdigest(), False
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    sas = getpass.getpass('Stanford SAS URL (hidden): ').strip()
    parsed, blobs = list_blobs(sas)
    print('Authorized test blobs:', len(blobs), 'bytes:', sum(item[1] for item in blobs), flush=True)
    progress_path = ROOT / 'download_progress.json'
    started = time.monotonic()
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, parsed, blob): blob[0] for blob in blobs}
        for future in concurrent.futures.as_completed(futures):
            name, size, sha, reused = future.result()
            results[name] = {'size': size, 'sha256': sha}
            if len(results) % 25 == 0 or len(results) == len(blobs):
                status = {'phase': 'downloading' if len(results) < len(blobs) else 'complete',
                          'images': len(results), 'total': len(blobs),
                          'fraction': len(results) / len(blobs),
                          'bytes_verified': sum(item['size'] for item in results.values()),
                          'elapsed_s': time.monotonic() - started}
                progress_path.write_text(json.dumps(status))
                print('PROGRESS', json.dumps(status), flush=True)
    manifest = {'source': 'Stanford AIMI CheXlocalize Azure blob container',
                'cohort': 'CheXpert test, 668 radiographs, 500 studies',
                'images': len(results), 'total_bytes': sum(item['size'] for item in results.values()),
                'groundtruth_sha256': json.loads((ROOT / 'manifest.json').read_text())['sha256'],
                'files': results}
    (ROOT / 'image_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('COMPLETE', len(results), 'images; no SAS URL stored', flush=True)


if __name__ == '__main__':
    main()
