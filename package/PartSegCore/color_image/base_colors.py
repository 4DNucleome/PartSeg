import typing

from PartSegCore.class_generator import BaseSerializableClass

from .color_data import inferno_data, magma_data, plasma_data, sitk_labels, viridis_data

Num = typing.Union[int, float]


class Color(BaseSerializableClass):
    """
    store color information

    :param red: red color value
    :param green: green color value
    :param blue: blue color value
    """

    # noinspection PyOverloads,PyMissingConstructor
    # pylint: disable=W0104
    @typing.overload
    def __init__(self, red: Num, green: Num, blue: Num):
        ...

    red: Num
    green: Num
    blue: Num

    def __hash__(self):
        return hash(("Color", self.red, self.green, self.blue))

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.red == other.red
            and self.green == other.green
            and self.blue == other.blue
        )

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Color(self.red + other, self.green + other, self.blue + other)
        elif isinstance(other, Color):
            return Color(self.red + other.red, self.green + other.green, self.blue + other.blue)
        raise ValueError(f"Type {type(other)} not supported")

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Color(self.red - other, self.green - other, self.blue - other)
        elif isinstance(other, Color):
            return Color(self.red - other.red, self.green - other.green, self.blue - other.blue)
        raise ValueError(f"Type {type(other)} not supported")


class ColorPosition(BaseSerializableClass):
    """
    :py:class`~.Color` with position. Position if from range [0,1]

    :param color_position: 0 means color for darkest pixel, 1 means value for brightest color.
        Linear interpolation between.
    :param color: Color
    """

    # noinspection PyOverloads,PyMissingConstructor
    @typing.overload
    def __init__(self, color_position: float, color: Color):
        ...  # pylint: disable=W0104

    color_position: float  # point in which this color is started to be used. value from range [0, 1]
    color: Color  # As name suggest. Color as RGB

    def __hash__(self):
        return hash(("ColorPosition", self.color_position)) ^ hash(self.color)

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and other.color_position == self.color_position
            and other.color == self.color
        )


ColorInfoType = typing.Tuple[typing.Tuple[Num, Num, Num], ...]


class BaseColormap:
    """Base class for all colormap representations. Define interface."""

    def bounds(self) -> typing.List[float]:
        """
        coordinates from scale [0-1]. For each value there is corresponding color (RGB)
        returned by :py:meth:`~.color_values`
        """
        raise NotImplementedError()

    def color_values(self) -> ColorInfoType:
        """
        RGB values for :py:meth:`bounds`
        """
        raise NotImplementedError()

    def get_points(self) -> typing.Iterable[float]:
        """
        return coordinates of interpolated points. Need have length 0, or
        :py:data:`PartSegCore.color_image.resolution`. Values from range [0-1].
        It is for future changes
        """
        raise NotImplementedError()

    def num_of_channels(self) -> int:
        """Num of channels (RGB) used by this colormap"""
        return len(self.non_zero_channels())

    def non_zero_channels(self) -> typing.List[int]:
        """return list with indices of nonzero channels"""
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()


class ColorMap(BaseSerializableClass, BaseColormap):
    colormap: typing.Tuple[ColorPosition, ...]
    points: typing.Tuple[float, ...] = ()

    def __getitem__(self, item):
        return self.colormap[item]

    def __iter__(self):
        return iter(self.colormap)

    def __len__(self):
        return len(self.colormap)

    def __hash__(self):
        # TODO (fixme to not need convert)
        return hash(tuple(self.colormap)) ^ hash(tuple(self.points))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.colormap == other.colormap and self.points == other.points

    def non_zero_channels(self) -> typing.List[int]:
        rm, gm, bm = 0, 0, 0
        for el in self.colormap:
            rm = max(rm, el.color.red)
            gm = max(gm, el.color.green)
            bm = max(bm, el.color.blue)
        res = list()
        if rm:
            res.append(0)
        if gm:
            res.append(1)
        if bm:
            res.append(2)
        return res

    def color_values(self) -> ColorInfoType:
        return tuple([x.color.as_tuple() for x in self.colormap])

    def bounds(self) -> typing.List[float]:
        return [x.color_position for x in self.colormap]

    def get_points(self) -> typing.Iterable[float]:
        return self.points


class ArrayColorMap(BaseColormap):
    def __init__(self, array: typing.Iterable, scale: float = 1):
        array = tuple([tuple([y * scale for y in x]) for x in array])
        if not all(map(lambda x: len(x) == 3, array)):
            raise ValueError("Colormaps should be defined as there channels (RGB)")
        self.array: ColorInfoType = array

    def bounds(self) -> typing.List[float]:
        arr_len = len(self.array)
        return [i / arr_len for i in range(arr_len)]

    def color_values(self) -> ColorInfoType:
        return self.array

    def __hash__(self):
        # TODO (fixme to not need convert)
        return hash(tuple(self.array))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.array == other.array

    def non_zero_channels(self) -> typing.List[int]:
        res = []
        for i in range(3):
            if max(self.array[i]) > 0:
                res.append(i)
        return res

    def get_points(self) -> typing.Iterable[float]:
        return []


def reverse_colormap(colormap: ColorMap) -> ColorMap:
    """reverse colormap"""
    return ColorMap(tuple([ColorPosition(1 - c.color_position, c.color) for c in reversed(colormap)]))


_black = Color(0, 0, 0)
_white = Color(255, 255, 255)

inferno = ArrayColorMap(inferno_data, 255)
inferno_r = ArrayColorMap(reversed(inferno_data), 255)
magma = ArrayColorMap(magma_data, 255)
magma_r = ArrayColorMap(reversed(magma_data), 255)
plasma = ArrayColorMap(plasma_data, 255)
plasma_r = ArrayColorMap(reversed(plasma_data), 255)
viridis = ArrayColorMap(viridis_data, 255)
viridis_r = ArrayColorMap(reversed(viridis_data), 255)


base_colors = [
    ("Red", Color(255, 0, 0)),
    ("Green", Color(0, 255, 0)),
    ("Blue", Color(0, 0, 255)),
    ("Magenta", Color(255, 0, 144)),
]

colormap_list = [
    ("Black" + name, ColorMap((ColorPosition(0, _black), ColorPosition(1, col)))) for name, col in base_colors
] + [("Grayscale", ColorMap((ColorPosition(0, _black), ColorPosition(1, _white))))]

colormap_list_r = [(x[0] + "_reversed", reverse_colormap(x[1])) for x in colormap_list]

colormap_list += [("inferno", inferno), ("magma", magma), ("plasma", plasma), ("viridis", viridis)]
colormap_list += colormap_list_r

colormap_list += [
    ("inferno_reversed", inferno_r),
    ("magma_reversed", magma_r),
    ("plasma_reversed", plasma_r),
    ("viridis_reversed", viridis_r),
]
# If changing this check ViewSettings.chosen_colormap

default_colormap_dict = {x[0]: x[1] for x in colormap_list}

starting_colors = [x[0] for x in colormap_list[:9]]

# print(default_colormap_dict)


default_label_dict = {"default": sitk_labels}

LabelColors = list
