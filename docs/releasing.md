# Releasing Actant

Actant publishes from a GitHub Release through PyPI trusted publishing. The
release workflow builds both the wheel and source distribution from the tagged
commit, verifies their metadata, and publishes only after the build succeeds.

## Version policy

Actant uses three-part, PEP 440-compatible semantic versions.

While the public API is pre-1.0:

- `0.Y.0` introduces features or intentionally breaks a public API;
- `0.Y.Z` contains backward-compatible fixes and small internal improvements;
- `0.Y.0rcN` is an optional release candidate for a change that needs external
  testing before its final release.

After `1.0.0`:

- increment **major** for incompatible public API changes;
- increment **minor** for backward-compatible features;
- increment **patch** for backward-compatible fixes.

Do not publish routine `.devN` builds to PyPI. Use commit SHAs for unreleased
development builds. Use `.postN` only to repair release metadata that cannot be
fixed with a normal patch release; code fixes always receive a new patch.

`pyproject.toml` is the single source of truth for the package version. Use
`uv version --bump patch`, `uv version --bump minor`, or an explicit PEP 440
version such as `uv version 0.2.0rc1`. The command also updates `uv.lock`.

## Changelog policy

`CHANGELOG.md` records user-facing changes. Add entries under `Unreleased` as
work lands. Before tagging:

1. replace `Unreleased` content with a dated version section;
2. restore an empty `Unreleased` section;
3. update its comparison link to start at the new tag;
4. use the same concise highlights in the GitHub Release notes.

GitHub Releases are the canonical distribution record; the checked-in
changelog makes upgrade history available without leaving the repository.

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
2. Choose the version according to the policy above. Update it with
   `uv version --bump patch`, `uv version --bump minor`, or
   `uv version <version>`.
3. Finalize that version's `CHANGELOG.md` section.
4. Review the generated `pyproject.toml`, `uv.lock`, and changelog changes.
5. Run the local release checks:

   ```bash
   just lint
   just typecheck
   just test
   just package
   ```

6. Commit and push the version bump; wait for CI.
7. Create a GitHub Release whose tag is exactly `v<version>` and targets that
   commit. For version `0.1.0`, the tag must be `v0.1.0`.
8. Copy the changelog highlights into the release notes and publish the GitHub
   Release.
9. Approve the `pypi` environment deployment if protection is enabled.
10. Verify the published artifacts and install from a fresh environment:

   ```bash
   uv venv /tmp/actant-release-check
   uv pip install --python /tmp/actant-release-check/bin/python actant
   /tmp/actant-release-check/bin/python -c \
     "from importlib.metadata import version; print(version('actant'))"
   ```

Do not rebuild or upload a release from a developer machine. PyPI files are
immutable; if a release is wrong, increment the version and publish a new one.
