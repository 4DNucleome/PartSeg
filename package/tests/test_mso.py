import pytest

from PartSeg.utils.multiscale_opening import PyMSO, calculate_mu, MuType
import numpy as np

from PartSeg.utils.segmentation.sprawl import NeighType, calculate_distances_array


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
        # TODO
        image = np.zeros((10, 10, 10), dtype=np.uint8)


def test_mso_construct():
    data = PyMSO()


class TestFDT:
    def test_fdt_base(self):
        mso = PyMSO()
        neigh, dist = calculate_distances_array((1, 1, 1), NeighType.vertex)
        components = np.zeros((10, 10, 10), dtype=np.uint8)
        components[3:7, 3:7, 3:7] = 1
        mso.set_neighbourhood(neigh, dist)
        mso.set_components(components)
        mso.set_mu_array(np.ones(components.shape))
        res = mso.calculate_FDT()
        print(res)

class TestExceptions:
    def test_fdt(self):
        mso = PyMSO()
        neigh, dist = calculate_distances_array((1, 1, 1), NeighType.vertex)
        components = np.zeros((10, 10, 10), dtype=np.uint8)
        components[3:7, 3:7, 3:7] = 1
        mso.set_neighbourhood(neigh, dist)
        with pytest.raises(RuntimeError):
            mso.calculate_FDT()
        mso.set_components(components)
        with pytest.raises(RuntimeError):
            mso.calculate_FDT()


