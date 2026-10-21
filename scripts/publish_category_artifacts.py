#!/usr/bin/env python3
"""Publish a prepared category bundle to new immutable R2 paths only."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from botocore.exceptions import ClientError

import chain_config
from teutonic.evaluation.index_manifest import DEFAULT_INDEX_MANIFEST_URL, pin_index_manifest
from teutonic.evaluation.mixes import (
    competition_snapshot,
    validate_definitions,
    version_manifest_url,
)


def publish_immutable(client, bucket, key, body, digest):
    """Never overwrite an existing object; verify exact bytes even on retry."""
    if hashlib.sha256(body).hexdigest() != digest:
        raise ValueError("local artifact checksum mismatch")
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
        try:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                IfNoneMatch="*",
                ContentType="application/json",
                Metadata={"sha256": digest},
            )
        except ClientError as race:
            if race.response["Error"]["Code"] != "PreconditionFailed":
                raise
    observed = hashlib.sha256()
    with client.get_object(Bucket=bucket, Key=key)["Body"] as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            observed.update(chunk)
    if observed.hexdigest() != digest:
        raise ValueError(f"immutable object checksum mismatch: {key}")


def publish_bundle(directory, bucket, index_file=None):
    import boto3

    validate_definitions()
    directory = Path(directory)
    entries = json.loads((directory / "bundle.json").read_text())["competitions"]
    if set(entries) != set(chain_config.COMPETITION_DEFAULTS):
        raise ValueError("publication needs all seven competitions")
    index_artifact = None
    if index_file:
        raw = Path(index_file).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        url = DEFAULT_INDEX_MANIFEST_URL.rsplit("/", 1)[0] + f"/manifest-{digest}.json"
        pin_index_manifest(raw, url)
        sources = json.loads((directory / "index-sources.json").read_text())["datasets"]
        if json.loads(raw)["datasets"] != sources:
            raise ValueError("document index source pins differ from the prepared bundle")
        index_artifact = (raw, digest, url)
    # Validate every local mix before making the first remote write.
    for key, entry in entries.items():
        snapshot = competition_snapshot(key, bundle_dir=directory)
        expected_url = version_manifest_url(key, snapshot.manifest_sha256)
        if entry["url"] != expected_url:
            raise ValueError("publication URL must be the content-addressed version path")
        if any(
            v["total_tokens"] != chain_config.CATEGORY_REGISTRY[c]["tokens"]
            for c, v in snapshot.manifest["category_inventory"].items()
        ):
            raise ValueError("token inventory changed; review category weights before publishing")
        del snapshot
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["TEUTONIC_R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ.get("R2_SESSION_TOKEN") or None,
        region_name="auto",
    )
    for key, entry in entries.items():
        publish_immutable(
            client,
            bucket,
            urlparse(entry["url"]).path.lstrip("/"),
            (directory / f"{key}.json").read_bytes(),
            entry["sha256"],
        )
        print(json.dumps({"verified": key, **entry}), flush=True)
    receipt = {"bucket": bucket, "competitions": entries}
    if index_artifact:
        raw, digest, url = index_artifact
        publish_immutable(client, bucket, urlparse(url).path.lstrip("/"), raw, digest)
        receipt["document_index"] = {"url": url, "sha256": digest}
    (directory / "publication.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--index-manifest-file", type=Path)
    parser.add_argument("--bucket", default=os.environ.get("TEUTONIC_DATASETS_BUCKET", "datasets"))
    args = parser.parse_args(argv)
    print(
        json.dumps(publish_bundle(args.bundle_dir, args.bucket, args.index_manifest_file), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
