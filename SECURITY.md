# Security Policy

## Checkpoint files

Checkpoint files (`checkpoint_merged.pkl`) use Python's `pickle` format
and should only be loaded from trusted local sources.  Never load a
checkpoint file received from an untrusted party — pickle deserialisation
can execute arbitrary code.

## Supported versions

Only the latest commit on the `main` branch is actively maintained.
