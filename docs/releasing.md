# Releasing Actant

Actant publishes from a GitHub Release through PyPI trusted publishing. The
release workflow builds both the wheel and source distribution from the tagged
commit, verifies their metadata, and publishes only after the build succeeds.

## One-time PyPI setup

Create the `actant` project on PyPI and configure a trusted publisher with:

- owner: `johnathanchiu`
- repository: `actant`
- workflow: `release.yml`
- environment: `pypi`

Create a protected GitHub environment named `pypi`. Requiring a reviewer there
adds a final manual gate between a successful build and publication.

## Release checklist

1. Confirm `main` is clean and the `ci / required` check is green.
2. Choose the version and update it with `uv version <version>`.
3. Review the generated `pyproject.toml` and `uv.lock` changes.
4. Run the local release checks:

   ```bash
   just lint
   just typecheck
   just test
   just package
   ```

5. Commit and push the version bump; wait for CI.
6. Create a GitHub Release whose tag is exactly `v<version>` and targets that
   commit. For version `0.1.0`, the tag must be `v0.1.0`.
7. Approve the `pypi` environment deployment if protection is enabled.
8. Verify the published artifacts and install from a fresh environment:

   ```bash
   uv venv /tmp/actant-release-check
   uv pip install --python /tmp/actant-release-check/bin/python actant
   /tmp/actant-release-check/bin/python -c \
     "from importlib.metadata import version; print(version('actant'))"
   ```

Do not rebuild or upload a release from a developer machine. PyPI files are
immutable; if a release is wrong, increment the version and publish a new one.
