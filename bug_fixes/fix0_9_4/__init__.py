import numpy as np

import PartSegImage


if PartSegImage.Image.__module__ == "PartSegImage.image":
    # noinspection DuplicatedCode
    def fit_mask_to_image(self, array: np.ndarray) -> np.ndarray:
        """call :py:meth:`fit_array_to_image` and then use minimal size type which save information"""
        array = self.fit_array_to_image(array)
        unique = np.unique(array)
        if unique.size == 2 and unique[1] == 1:
            return array.astype(np.uint8)
        if unique.size == 1:
            if unique[0] != 0:
                return np.ones(array.shape, dtype=np.uint8)
            return array.astype(np.uint8)
        max_val = unique.max()
        if max_val + 1 == unique.size:
            if max_val < 250:
                return array.astype(np.uint8)
            else:
                return array.astype(np.uint32)
        masking_array = np.zeros(max_val + 1, dtype=np.uint32)
        for i, val in enumerate(unique, 0 if unique[0] == 0 else 1):
            masking_array[val] = i
        res = masking_array[array]
        if len(unique) < 250:
            return res.astype(np.uint8)
        return res

    PartSegImage.Image.fit_mask_to_image = fit_mask_to_image
