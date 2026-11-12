# PCNN

An experimental Python implementation of a unit-linking pulse-coupled neural
network. See [the publication review](doc/publication-review.md) for the distinction
between this pulse generator and the published segmentation algorithm.

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run main.py path/to/image.png
# A folder of images is also accepted.
uv run python -m pytest
```

Results and GIFs are written to `out/`. Existing files with matching output names
are overwritten; other files are preserved. OpenCV is installed without GUI
support; the code does not use OpenCV windows.

To use the existing notebooks:

```sh
uv sync --locked --group notebooks
uv run --group notebooks jupyter lab
```

Runtime dependencies and optional notebook/development dependencies are declared
in `pyproject.toml`; `uv.lock` records the resolved versions. Use `uv lock --upgrade`
and rerun the tests when updating them.
