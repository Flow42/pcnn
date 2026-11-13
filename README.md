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

Launch the interactive Gradio visualization:

```sh
uv sync --locked
uv run python -m visualization.app
# Optional: --host 0.0.0.0 --port 7860
```

Open `http://127.0.0.1:7860`, upload an image, configure the PCNN, and select
**Run algorithm**. The Material Design styled interface includes an MP4 player
with download controls, a frame inspector with a slider and previous/next buttons,
and the highest-entropy frame. All frames are retained, including repeated frames.
The frame inspector shows the original binary pixels; the H.264 video is compressed
and padded by one pixel on the bottom/right when dimensions are odd.

Controls cover iterations, threshold decay (α), linking strength (β), threshold
increment (V), initial threshold, height/width downsampling, and illumination
normalization. Parameters appear after uploading an image and wrap into columns
as space allows. The upload and settings cards sit above the results. Playback
uses 8 fps, a 7 × 7 neighborhood, and no dropout; these defaults remain available
to algorithm adapters without exposing controls in the UI.
Images are converted to grayscale and min-max normalized as in the command-line
algorithm. Inputs are limited to 16 megapixels and runs to 64 million stored pixels;
the frame limit is a viewer memory safeguard, not a PCNN limitation. At this limit,
raw frames take about 64 MB, with additional memory needed for processing and
export. Increase downsampling or reduce iterations for larger images. Temporary
videos are removed on rerun or session expiry; Gradio's served file cache expires
after an hour.

The viewer is split into `visualization/algorithms.py` (algorithm protocol, parameter
schema, and PCNN adapter), `visualization/processing.py` (frame collection, entropy
selection, and video export), and `visualization/app.py` (Gradio controls and events).
To add another algorithm, implement `name`, `description`, `parameters`, and
`frames(image_path, values)`, then register it in `ALGORITHMS`. Frames must be
same-sized, nonempty 2D `uint8` arrays. Controls are generated from each adapter's
parameter schema; playback and navigation are shared. The original CLI is unchanged.

To use the existing notebooks:

```sh
uv sync --locked --group notebooks
uv run --group notebooks jupyter lab
```

Runtime dependencies and optional notebook/development dependencies are declared
in `pyproject.toml`; `uv.lock` records the resolved versions. Use `uv lock --upgrade`
and rerun the tests when updating them.
