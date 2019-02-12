from PartSeg.utils.multiscale_opening import PyMSO, calculate_mu, MuType
import numpy as np


def test_mso_construct():
    data = PyMSO()


def test_base_mu():
    image = np.zeros((10, 10, 10), dtype=np.uint8)
    image[2:8, 2:8, 2:8] = 10
    res = calculate_mu(image, 2, 8, MuType.base_mu)
    assert np.all(res == (image > 0).astype(np.float64))
    res = calculate_mu(image, 5, 15, MuType.base_mu)
    assert np.all(res == (image > 0).astype(np.float64) * 0.5)
    image[4:6, 4:6, 4:6] = 20
    res = calculate_mu(image, 5, 15, MuType.base_mu)
    assert np.all(res == ( (image > 0).astype(np.float64) + (image > 15).astype(np.float64)) * 0.5)


def test_reversed_base_mu():
    image = np.zeros((10, 10, 10), dtype=np.uint8)
    image[2:8, 2:8, 2:8] = 10
    res = calculate_mu(image, 8, 2, MuType.base_mu)
    assert np.all(res == (image == 0).astype(np.float64))
    res = calculate_mu(image, 15, 5, MuType.base_mu)
    assert np.all(res == ((20 - image)/20).astype(np.float64))
    image[4:6, 4:6, 4:6] = 20
    res = calculate_mu(image, 15, 5, MuType.base_mu)
    assert np.all(res == (20 - image)/20)
