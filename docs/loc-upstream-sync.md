# LOC upstream sync

This repository tracks the original LOC code as:

```bash
loc-upstream  https://github.com/mjuvela/LOC.git
```

Use `loc-upstream` for fetching upstream LOC updates. Push local elmod work to
`origin`, not to `loc-upstream`.

## Local patch areas

Local LOC-related changes are intentionally concentrated in:

- `example/LOC1D.py`
- `example/LOC_aux.py`
- `example/loc_worker.py`
- `module.py`
- `misc.py`
- example configuration in `example/oph_464_n2hp_HFS.ini`

The main local changes are:

- persistent LOC worker execution instead of starting `LOC1D.py` for every likelihood
- OpenCL context, program, and kernel reuse
- quieter fitting/convolution output, with `ELMOD_VERBOSE=1` for debug output
- robust likelihood/prior handling and walker validation
- LOC stopping by maximum relative level-population change

## Fetch upstream

```bash
git fetch loc-upstream
git log --oneline --decorate -10 loc-upstream/main
```

## Inspect upstream changes

Check what changed upstream before applying anything:

```bash
git diff --stat main..loc-upstream/main
git diff main..loc-upstream/main -- example/LOC1D.py example/LOC_aux.py
```

If the upstream project edits files we patch locally, expect conflicts.

## Apply upstream updates

Work on a temporary sync branch so `main` stays usable:

```bash
git switch -c codex/sync-loc-upstream
git rebase loc-upstream/main
```

Resolve conflicts by preserving both:

- upstream LOC bug fixes and scientific changes
- local elmod integration/performance fixes listed above

After resolving conflicts:

```bash
python3 -m py_compile example/LOC1D.py example/LOC_aux.py example/loc_worker.py module.py misc.py
```

Then run the example smoke test on a machine with the needed OpenCL/LOC inputs:

```bash
cd example
python run.py
```

When the sync branch is good, merge it back into `main` or open a pull request,
then push to `origin`.

## Abort a bad sync

If a rebase becomes messy:

```bash
git rebase --abort
git switch main
```

Then start again from a fresh sync branch.
