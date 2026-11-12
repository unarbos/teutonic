-- Local migration artifact. Apply only during the approved, coordinated cutover.
-- Existing manifests and evaluation history are preserved.
BEGIN;
ALTER TABLE control_plane.dataset_manifests
    DROP CONSTRAINT dataset_manifests_manifest_url_check;
ALTER TABLE control_plane.dataset_manifests
    ADD CONSTRAINT dataset_manifests_manifest_url_check CHECK (
        manifest_url ~ '^https://' OR
        manifest_url ~ '^s3://dataset/teutonic-text/splits/[a-z0-9_-]+/[0-9a-f]{64}\.json$'
    );
COMMIT;
