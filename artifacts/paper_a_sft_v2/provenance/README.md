# Clean-v2 execution-source snapshot

`paper-a-v2-execution-source-b5f491f.tar.zst` is a tracked-tree snapshot of
the exact source commit used by the isolated GPU run. It is intentionally not a
full-history Git bundle: the lock already binds every scientific execution file,
and excluding history reduces the unrelated release surface.

Verify the archive checksum before extracting it. After extraction, recompute the
16 file hashes listed in `../LOCK.json` under `execution_sources.files`; their
canonical aggregate must equal the lock's `execution_sources.aggregate_sha256`.
The post-run checkout contains additional analysis and release hardening, so its
source bytes are expected to differ and are attested separately by generated
analysis metadata.

The snapshot was audited for archive traversal entries, symlinks, device files,
credentials, private keys, authenticated URLs, and raw Paper A prompt rows. None
were found. The commit is unsigned, so the SHA-256 checksum, commit ID, tree ID,
and lock-bound per-file hashes are all retained in `execution-source-snapshot.json`.

`gcp-runner.sh` and `paper-a-v2.service` preserve the exact fail-closed stage
orchestration and service definition used on the ephemeral VM. They contain no
credentials; authentication was provided only through the VM service account and
the untracked runtime `.env` file.

`execution-evidence.json` records the independently checked cloud-archive hash,
row/bundle/adapter counts, repeat-analysis result, exact-source verification,
release-contract digests, and post-run GCP cleanup state. The 4.43 GB full cloud
archive is not committed to Git; its outer digest and all 177 internal checksums
were verified after download. The compact execution-source snapshot above is the
durable tracked source witness.
