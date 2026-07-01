# Contributing to Agent Bouncer

Thanks for your interest! This project aims to be a **crystal-clear, reproducible**
guardrail repo, so contributions that improve clarity, tests, and benchmark
coverage are especially welcome.

## Development workflow

We use **branch → PR → merge**. Never commit directly to `main`.

1. Create a branch: `git checkout -b feat/<short-name>` (or `fix/`, `docs/`, `chore/`).
2. Make your change with tests.
3. Run the local quality gate:
   ```bash
   make lint && make test
   ```
4. Open a Pull Request against `main` and fill in the template.
5. A maintainer reviews and merges. `main` is protected.

## Getting set up

```bash
make setup            # venv + dev/eval extras
make test             # run tests
make eval             # smoke-test the eval harness on the reference guard
```

## Ground rules

- Keep the **core package dependency-light** (heavy ML libs live behind extras).
- Every new guard implements the `Guard` protocol and returns a `Verdict`.
- New metrics/rewards must ship with unit tests.
- Datasets must be license-compatible (Apache-2.0 preferred — see `docs/datasets.md`).

## Responsible use

This is a **defensive** security tool. Do not submit content or features whose
primary purpose is to cause harm. See `SECURITY.md` for reporting vulnerabilities.
