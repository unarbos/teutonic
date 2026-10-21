#!/usr/bin/env python3
"""Build a local document-index inventory. Reads R2; never uploads.

Use original source inventories or a legacy MAIN config to enumerate index objects.
Metadata-only mode uses HEAD requests and preserves existing verified checksums.
SHA-256 can also be computed from local originals (--index-root). This command
never downloads index bodies to hash them.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teutonic.evaluation.index_manifest import (
    ETAG_INDEX_FORMAT,
    INDEX_FORMAT,
    TOKEN_FORMAT,
    pin_index_manifest,
)
from teutonic.evaluation.long_documents import describe_index_file


def hash_remote_ranges(descriptor, workers, *, chunk_bytes=32 * 1024 * 1024, opener=urlopen):
    """Hash in file order, with bounded concurrent range reads and bounded RAM."""
    size = descriptor['size_bytes']
    def read(start):
        end = min(size, start + chunk_bytes) - 1
        headers = {'User-Agent': 'teutonic-eval/1.0', 'If-Match': descriptor['etag'],
                   'Accept-Encoding': 'identity', 'Range': f'bytes={start}-{end}'}
        with opener(Request(descriptor['url'], headers=headers), timeout=120) as response:
            if response.status != 206 or response.headers.get('ETag') != descriptor['etag'] or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                raise ValueError('index changed or hash range was not honored')
            data = response.read(end-start+2)
        if len(data) != end-start+1:
            raise ValueError('truncated index hash range')
        return data
    digest = hashlib.sha256()
    offsets = iter(range(0, size, chunk_bytes))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(read, start) for start in [next(offsets, None) for _ in range(workers)] if start is not None]
        while pending:
            digest.update(pending.pop(0).result())
            start = next(offsets, None)
            if start is not None:
                pending.append(pool.submit(read, start))
    return digest.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="legacy prepared long-document config")
    group.add_argument("--sources", type=Path, help="index-sources.json from prepare_competition_mixes.py")
    p.add_argument("--base-manifest", type=Path, help="reuse unchanged, verified index descriptors")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest-url", required=True)
    p.add_argument("--index-root", type=Path)
    p.add_argument("--metadata-only", action="store_true", help="use R2 ETag and size pins; no index-body downloads or hashing")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    if not args.metadata_only and not args.index_root:
        p.error("choose --metadata-only or --index-root; remote hashing is disabled")
    if args.metadata_only and args.index_root:
        p.error("choose only one of --metadata-only or --index-root")
    from teutonic.evaluation.index_manifest import index_paths, validate_inventory
    prefix = args.manifest_url.rsplit('/', 1)[0] + '/'
    objects = {}
    datasets = {}
    base = json.loads(args.base_manifest.read_text()) if args.base_manifest else None
    if base:
        validate_inventory(base, args.manifest_url)
    if args.sources:
        datasets = json.loads(args.sources.read_text())["datasets"]
        for name, entry in datasets.items():
            for category in entry["categories"]:
                for path in index_paths(name, category).values():
                    objects[path] = describe_index_file(prefix + path)
    else:
        config = json.loads(args.config.read_text())
        sources = {s['name']: {k: s[k] for k in ('url', 'sha256')} for s in config['manifests']}
        for category in config['categories']:
            name = category['dataset']
            entry = datasets.setdefault(name, {'categories': [], 'source_manifest': sources[name]})
            entry['categories'].append(category['category'])
            for descriptor in category['files'].values():
                if not descriptor['url'].startswith(prefix):
                    raise ValueError('index files must be under the manifest directory')
                path = descriptor['url'][len(prefix):]
                if '..' in path.split('/') or urlparse(path).query:
                    raise ValueError('unsafe index path')
                objects[path] = descriptor
    for dataset in datasets.values():
        dataset['categories'] = sorted(set(dataset['categories']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipts = args.output.parent / 'index-hash-receipts'
    receipts.mkdir(parents=True, exist_ok=True)

    def hash_object(entry):
        path, expected = entry
        observed = describe_index_file(expected['url'])
        if any(observed[k] != expected[k] for k in ('url', 'etag', 'size_bytes')):
            raise ValueError(f'index changed: {path}')
        if base and path in base['files']:
            old = base['files'][path]
            dataset = path.split('/', 1)[0]
            if (base['datasets'].get(dataset) == datasets.get(dataset)
                and all(old[k] == observed[k] for k in ('etag', 'size_bytes'))):
                print(f'reused {path}', flush=True)
                return path, old
        receipt = receipts / (hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest() + '.json')
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value['object'] == observed:
                print(f'cached {path}', flush=True)
                return path, {k: value[k] for k in ('sha256', 'size_bytes', 'etag')}
        if args.metadata_only:
            value = {k: observed[k] for k in ('size_bytes', 'etag')}
            print(f'metadata-pinned {path}', flush=True)
            return path, value
        digest, count = hashlib.sha256(), 0
        stream = (args.index_root / path).open('rb')
        with stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
                count += len(block)
        if count != expected['size_bytes']:
            raise ValueError(f'index size mismatch: {path}')
        value = {'sha256': digest.hexdigest(), 'size_bytes': count, 'etag': expected['etag']}
        temp = receipt.with_suffix('.tmp')
        temp.write_text(json.dumps({'object': observed, **value}))
        temp.replace(receipt)
        print(json.dumps({'hashed': path, **value}), flush=True)
        return path, value

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = dict(pool.map(hash_object, sorted(objects.items(), key=lambda x: x[1]['size_bytes'])))
    manifest = {'format': ETAG_INDEX_FORMAT if any('sha256' not in f for f in files.values()) else INDEX_FORMAT,
                'token_format': TOKEN_FORMAT, 'datasets': datasets, 'files': files}
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    pin = pin_index_manifest(raw, args.manifest_url)
    args.output.write_bytes(raw)
    args.output.with_suffix('.json.sha256').write_text(pin['sha256'] + '  ' + args.output.name + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': pin['sha256'], 'files': len(files), 'size_bytes': sum(f['size_bytes'] for f in files.values())}), flush=True)


if __name__ == '__main__':
    main()
