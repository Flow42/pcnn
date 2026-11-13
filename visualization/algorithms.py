"""UI-independent algorithm contract and registry.

Adapters accept an uploaded image path and parameter values, and yield uint8
grayscale frames. Add an adapter to ALGORITHMS to expose its controls in the UI.
"""

from dataclasses import dataclass
from typing import Iterator, Mapping, Protocol

import numpy as np

from pcnn import pcnn
from pcnn_util import read_img


@dataclass(frozen=True)
class Parameter:
    name: str
    label: str
    default: float | int | bool
    minimum: float = 0
    maximum: float = 1
    step: float = 0.01
    info: str = ""
    visible: bool = True


class Algorithm(Protocol):
    name: str
    description: str
    parameters: tuple[Parameter, ...]

    def frames(self, image_path: str, values: Mapping) -> Iterator[np.ndarray]: ...


class PCNNAlgorithm:
    name = "Unit-linking PCNN"
    description = "A pulse-coupled neural network. Each frame shows the neurons firing during one iteration."
    parameters = (
        Parameter("epochs", "Iterations", 20, 1, 300, 1),
        Parameter("alpha", "Threshold decay · α", 0.05, 0, 1, 0.005),
        Parameter("beta", "Linking strength · β", 1.0, 0, 10, 0.05),
        Parameter("V", "Threshold increment · V", 0.99, 0, 5, 0.01),
        Parameter("kernel", "Neighborhood size", 7, 1, 31, 2,
                  "Odd square kernel; the center does not link to itself.", visible=False),
        Parameter("threshold", "Initial threshold", 1.0, 0, 5, 0.01),
        Parameter("scale_height", "Height downsampling factor", 1, 1, 16, 1),
        Parameter("scale_width", "Width downsampling factor", 1, 1, 16, 1),
        Parameter("brightness_is_normed", "Normalize illumination", False, info="Ratio-of-Gaussians brightness normalization."),
        Parameter("dropout", "Random neighborhood dropout", False,
                  info="Randomly disables kernel entries each iteration; runs can differ.", visible=False),
    )

    def frames(self, image_path, values):
        params = {p.name: values.get(p.name, p.default) for p in self.parameters}
        for parameter in self.parameters:
            value = params[parameter.name]
            if isinstance(parameter.default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{parameter.label} must be a boolean.")
            elif (not np.isfinite(value) or not parameter.minimum <= value <= parameter.maximum
                  or (isinstance(parameter.default, int) and int(value) != value)):
                raise ValueError(f"Invalid value for {parameter.label}.")
        if int(params["kernel"]) % 2 == 0:
            raise ValueError("Neighborhood size must be odd.")
        # Check dimensions before decoding and allocating a full-resolution image.
        from PIL import Image
        with Image.open(image_path) as source:
            if source.width * source.height > 16_000_000:
                raise ValueError("Please upload an image with at most 16 megapixels.")
        image = read_img(
            image_path,
            scales=(int(params["scale_height"]), int(params["scale_width"])),
            brightness_is_normed=params["brightness_is_normed"],
        )
        if image.size * int(params["epochs"]) > 64_000_000:
            raise ValueError(
                "This run exceeds the viewer's memory budget of 64 million stored pixels "
                "(about 64 MB of raw frames, plus processing overhead). "
                "Increase downsampling or reduce iterations. This is a viewer safeguard, not a PCNN limit."
            )
        model = pcnn(
            kernel=int(params["kernel"]), epochs=int(params["epochs"]),
            alpha=float(params["alpha"]), beta=float(params["beta"]), V=float(params["V"]),
            dropout=params["dropout"],
        )
        generator = model.gray_pcnn_gen(image, thresh=float(params["threshold"]))
        for _ in range(model.epochs):
            yield next(generator).astype(np.uint8) * 255


ALGORITHMS: dict[str, Algorithm] = {"pcnn": PCNNAlgorithm()}
