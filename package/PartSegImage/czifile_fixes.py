import inspect
import typing
from importlib import metadata

import numpy as np
from packaging.version import parse as parse_version

if parse_version(metadata.version("czifile")) < parse_version("2026.3.12"):
    import imagecodecs
    from czifile.czifile import DECOMPRESS

    class ZSTD1Header(typing.NamedTuple):
        """
        ZSTD1 header structure
        based on:
        https://github.com/ZEISS/libczi/blob/4a60e22200cbf0c8ff2a59f69a81ef1b2b89bf4f/Src/libCZI/decoder_zstd.cpp#L19
        """

        header_size: int
        hiLoByteUnpackPreprocessing: bool

    def parse_zstd1_header(data: bytes, size: int) -> ZSTD1Header:  # pragma: no cover
        """
        Parse ZSTD header

        https://github.com/ZEISS/libczi/blob/4a60e22200cbf0c8ff2a59f69a81ef1b2b89bf4f/Src/libCZI/decoder_zstd.cpp#L84
        """
        if size < 1:
            return ZSTD1Header(0, False)

        if data[0] == 1:
            return ZSTD1Header(1, False)

        if data[0] == 3 and size < 3:
            return ZSTD1Header(0, False)

        if data[1] == 1:
            return ZSTD1Header(3, bool(data[2] & 1))

        return ZSTD1Header(0, False)

    def _get_dtype():
        return inspect.currentframe().f_back.f_back.f_locals["de"].dtype

    def decode_zstd1(data: bytes) -> np.ndarray:
        """
        Decode ZSTD1 data
        """
        header = parse_zstd1_header(data, len(data))
        dtype = _get_dtype()
        if header.hiLoByteUnpackPreprocessing:
            array_ = np.frombuffer(imagecodecs.zstd_decode(data[header.header_size :]), np.uint8).copy()
            half_size = array_.size // 2
            array = np.empty(half_size, np.uint16)
            array[:] = array_[:half_size] + (array_[half_size:].astype(np.uint16) << 8)
            array = array.view(dtype)
        else:
            array = np.frombuffer(imagecodecs.zstd_decode(data[header.header_size :]), dtype).copy()
        return array

    def decode_zstd0(data: bytes) -> np.ndarray:
        """
        Decode ZSTD0 data
        """
        dtype = _get_dtype()
        return np.frombuffer(imagecodecs.zstd_decode(data), dtype).copy()

    DECOMPRESS[5] = decode_zstd0
    DECOMPRESS[6] = decode_zstd1
