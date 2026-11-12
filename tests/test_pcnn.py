import cv2
import numpy as np
import pytest
from pcnn import pcnn
from pcnn_util import read_img, normalize_brightness


def test_equations_and_no_self_link():
    image = np.array([[0.8, 0.6, 0.1]])
    model = pcnn(kernel=3, alpha=0.05, beta=0.2, V=1)
    gen = model.gray_pcnn_gen(image, Y_init=np.array([[1., 0., 0.]]), thresh=0.7)
    np.testing.assert_array_equal(next(gen), [[True, True, False]])
    np.testing.assert_array_equal(next(gen), [[False, False, False]])
    gen = model.gray_pcnn_gen(np.array([[0.6]]), Y_init=np.ones((1, 1)), thresh=0.7)
    assert not next(gen).any()


def test_entropy_non_square_and_uniform():
    model = pcnn(epochs=3)
    masks = [np.zeros((2, 3), bool), np.array([[1, 1, 1], [0, 0, 0]], bool), np.ones((2, 3), bool)]
    model.gray_pcnn_gen = lambda *args: iter(masks)
    with np.errstate(all='raise'):
        np.testing.assert_array_equal(model.gray_pcnn(np.zeros((2, 3))), masks[1])
    assert not pcnn(epochs=1).gray_pcnn(np.zeros((2, 3))).any()


def test_image_io_and_output(tmp_path):
    path = tmp_path / 'input.png'
    cv2.imwrite(str(path), np.array([[0, 100], [150, 255]], np.uint8))
    model = pcnn(epochs=3, write_gif=True, out_path=str(tmp_path / 'out'))
    result = model.run(str(path))
    assert result[0].shape == (2, 2)
    assert (tmp_path / 'out' / 'input.gif').exists()
    marker = tmp_path / 'out' / 'keep.txt'
    marker.write_text('keep')
    model.run(str(path))
    assert marker.read_text() == 'keep'
    assert np.isfinite(normalize_brightness(np.zeros((2, 2)))).all()
    with pytest.raises(ValueError, match='Cannot read'):
        read_img(tmp_path / 'missing.png')
