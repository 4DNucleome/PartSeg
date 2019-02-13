from PartSeg.utils.multiscale_opening import PyMSO, calculate_mu, MuType
import numpy as np


def test_mso_construct():
    data = PyMSO()


class TestMu:
    def test_base_mu(self):
        image = np.zeros((10, 10, 10), dtype=np.uint8)
        image[2:8, 2:8, 2:8] = 10
        res = calculate_mu(image, 2, 8, MuType.base_mu)
        assert np.all(res == (image > 0).astype(np.float64))
        res = calculate_mu(image, 5, 15, MuType.base_mu)
        assert np.all(res == (image > 0).astype(np.float64) * 0.5)
        image[4:6, 4:6, 4:6] = 20
        res = calculate_mu(image, 5, 15, MuType.base_mu)
        assert np.all(res == ((image > 0).astype(np.float64) + (image > 15).astype(np.float64)) * 0.5)

    def test_base_mu_masked(self):
        image = np.zeros((10, 10, 10), dtype=np.uint8)
        image[2:8, 2:8, 2:8] = 10
        res = calculate_mu(image, 2, 8, MuType.base_mu, image > 0)
        assert np.all(res == (image > 0).astype(np.float64))
        res = calculate_mu(image, 5, 15, MuType.base_mu, image > 0)
        assert np.all(res == (image > 0).astype(np.float64) * 0.5)
        image[4:6, 4:6, 4:6] = 20
        res = calculate_mu(image, 5, 15, MuType.base_mu, image > 0)
        assert np.all(res == ((image > 0).astype(np.float64) + (image > 15).astype(np.float64)) * 0.5)

    def test_reversed_base_mu(self):
        image = np.zeros((10, 10, 10), dtype=np.uint8)
        image[2:8, 2:8, 2:8] = 10
        res = calculate_mu(image, 8, 2, MuType.base_mu)
        assert np.all(res == (image == 0).astype(np.float64))
        res = calculate_mu(image, 15, 5, MuType.base_mu)
        assert np.all(res == ((20 - image)/20).astype(np.float64))
        image[4:6, 4:6, 4:6] = 20
        res = calculate_mu(image, 15, 5, MuType.base_mu)
        assert np.all(res == (20 - image)/20)

    def test_reversed_base_mu_masked(self):
        image = np.zeros((10, 10, 10), dtype=np.uint8)
        image[2:8, 2:8, 2:8] = 10
        mask = image > 0
        res = calculate_mu(image, 8, 2, MuType.base_mu, mask)
        assert np.all(res == np.zeros(image.shape, dtype=np.float64))
        res = calculate_mu(image, 15, 5, MuType.base_mu, mask)
        assert np.all(res == mask * 0.5)
        image[4:6, 4:6, 4:6] = 20
        res = calculate_mu(image, 15, 5, MuType.base_mu, mask)
        assert np.all(res == (mask * (image < 20)) * 0.5)

    def test_reflection_mu(self):
        image = np.zeros((10, 10, 10), dtype=np.uint8)
        image[2:8, 2:8, 2:8] = 10
        res = calculate_mu(image, 2, 8, MuType.reflection_mu)
        assert np.all(res == np.ones(image.shape, dtype=np.float64))
        res = calculate_mu(image, 5, 15, MuType.reflection_mu)
        assert np.all(res == (20 - image) / 20)
        image[4:6, 4:6, 4:6] = 20
        res = calculate_mu(image, 5, 15, MuType.reflection_mu)
        assert np.all(res == np.ones(image.shape, dtype=np.float64) - (image == 10) * 0.5)

    def test_reflection_mu_masked(self):
        image = np.zeros((10, 10, 10), dtype=np.uint8)
        image[2:8, 2:8, 2:8] = 10
        mask = image > 0
        res = calculate_mu(image, 2, 8, MuType.reflection_mu, mask)
        assert np.all(res == mask * 1.0)
        res = calculate_mu(image, 5, 15, MuType.reflection_mu, mask)
        assert np.all(res == ((20 - image) / 20) * mask)
        image[4:6, 4:6, 4:6] = 20
        res = calculate_mu(image, 5, 15, MuType.reflection_mu, mask)
        assert np.all(res == (np.ones(image.shape, dtype=np.float64) - (image == 10) * 0.5) * mask)

    def test_reversed_reflection_mu(self):
        image = np.zeros((10, 10, 10), dtype=np.uint8)
        image[2:8, 2:8, 2:8] = 10
        res = calculate_mu(image, 8, 2, MuType.reflection_mu)
        assert np.all(res == np.ones(image.shape, dtype=np.float64))
        res = calculate_mu(image, 15, 5, MuType.reflection_mu)
        assert np.all(res == (20 - image) / 20)
        image[4:6, 4:6, 4:6] = 20
        res = calculate_mu(image, 15, 5, MuType.reflection_mu)
        assert np.all(res == np.ones(image.shape, dtype=np.float64) - (image == 10) * 0.5)

    def test_reversed_reflection_mu_masked(self):
        image = np.zeros((10, 10, 10), dtype=np.uint8)
        image[2:8, 2:8, 2:8] = 10
        mask = image > 0
        res = calculate_mu(image, 8, 2, MuType.reflection_mu, mask)
        assert np.all(res == mask * 1.0)
        res = calculate_mu(image, 15, 5, MuType.reflection_mu, mask)
        assert np.all(res == ((20 - image) / 20) * mask)
        image[4:6, 4:6, 4:6] = 20
        res = calculate_mu(image, 15, 5, MuType.reflection_mu, mask)
        assert np.all(res == (np.ones(image.shape, dtype=np.float64) - (image == 10) * 0.5) * mask)

    def test_two_object_mu(self):
        #TODO
        image = np.zeros((10, 10, 10), dtype=np.uint8)
