import asyncio
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pytest

from pcnn import pcnn
from pcnn_util import read_img
from visualization.algorithms import ALGORITHMS, Parameter
from visualization.processing import cleanup_result, inspect_frame, process_image


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "input.png"
    cv2.imwrite(str(path), np.arange(35, dtype=np.uint8).reshape(5, 7) * 7)
    return str(path)


def test_adapter_matches_pcnn_and_video_round_trip(image_path):
    result = process_image(ALGORITHMS["pcnn"], image_path, {"epochs": 4}, fps=5)
    try:
        model = pcnn(kernel=7, epochs=4, alpha=0.05, beta=1.0, V=0.99)
        generator = model.gray_pcnn_gen(read_img(image_path), thresh=1.0)
        for frame in result.frames:
            np.testing.assert_array_equal(frame, next(generator).astype(np.uint8) * 255)
        np.testing.assert_array_equal(result.frames[result.best_index], model.gray_pcnn(read_img(image_path)) * 255)
        reader = imageio.get_reader(result.video_path)
        try:
            assert reader.count_frames() == 4
            assert reader.get_data(0).shape == (6, 8, 3)
            assert reader.get_meta_data()["fps"] == 5
        finally:
            reader.close()
        assert inspect_frame(result, -20)[1] == 1
        assert inspect_frame(result, 99)[1] == 4
        assert "0.60 s" in inspect_frame(result, 4)[2]
    finally:
        cleanup_result(result)
    assert not Path(result.video_path).parent.exists()


def test_input_validation(image_path):
    with pytest.raises(ValueError, match="Upload"):
        process_image(ALGORITHMS["pcnn"], "", {})
    with pytest.raises(ValueError, match="odd"):
        process_image(ALGORITHMS["pcnn"], image_path, {"kernel": 4})
    with pytest.raises(ValueError, match="Iterations"):
        process_image(ALGORITHMS["pcnn"], image_path, {"epochs": 0})
    assert inspect_frame(None, 1)[0] is None


def test_alternative_algorithm_and_single_frame(image_path):
    class Alternate:
        name = "Alternate"
        description = "Test algorithm"
        parameters = (Parameter("level", "Level", 1, 0, 2, 1),)

        def frames(self, image_path, values):
            yield np.full((2, 2), values["level"], dtype=np.uint8)

    result = process_image(Alternate(), image_path, {"level": 2})
    try:
        assert result.best_index == 0
        assert inspect_frame(result, 2)[1] == 1
    finally:
        cleanup_result(result)


def test_gradio_dynamic_controls_and_run(image_path):
    from gradio.state_holder import SessionState
    from visualization.app import build_app, material_theme

    app = build_app()
    material_theme()
    state = SessionState(app)
    render_fn = next(fn for fn in app.fns.values() if fn.renderable is not None)

    async def render_before_and_after_upload():
        await app.process_api(render_fn, ["pcnn", None], state=state,
                              session_hash="test")
        assert not any(fn.name == "run_algorithm" for fn in state.blocks_config.fns.values())
        uploaded = {"path": image_path, "meta": {"_type": "gradio.FileData"}}
        await app.process_api(render_fn, ["pcnn", uploaded], state=state,
                              session_hash="test", explicit_call=True)

    asyncio.run(render_before_and_after_upload())
    run_fn = next(fn for fn in state.blocks_config.fns.values() if fn.name == "run_algorithm")
    # Dynamic event inputs need explicit interactivity: Gradio cannot infer
    # their editable state reliably while serializing the rendered controls.
    editable_inputs = [run_fn.inputs[0], *run_fn.inputs[2:]]
    assert all(component.interactive is True for component in editable_inputs)
    labels = {component.label for component in editable_inputs}
    assert not labels.intersection({"Video speed (fps)", "Neighborhood size", "Random neighborhood dropout"})
    assert "Iterations" in labels
    dropdown = next(component for component in app.blocks.values()
                    if getattr(component, "label", None) == "Algorithm")
    assert dropdown.interactive is True
    values = [p.default for p in ALGORITHMS["pcnn"].parameters if p.visible]
    outputs = run_fn.fn(image_path, None, *values)
    try:
        assert outputs[3].maximum == 20
        assert "20 frames" in outputs[6]
        assert Path(outputs[1]).is_file()
    finally:
        cleanup_result(outputs[0])
    values[0] = 1
    single = run_fn.fn(image_path, None, *values)
    try:
        assert single[3].interactive is False
        assert single[7].interactive is False
        assert single[8].interactive is False
    finally:
        cleanup_result(single[0])
