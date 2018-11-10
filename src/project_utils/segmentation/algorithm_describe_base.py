import typing


class AlgorithmProperty(object):
    """
    :type name: str
    :type value_type: type
    :type default_value: typing.Union[object, str, int, float]
    """

    def __init__(self, name: str, user_name: str, default_value, options_range=None, single_steep=None,
                 possible_values=None, property_type=None):
        self.name = name
        self.user_name = user_name
        if type(options_range) is list:
            self.value_type = list
        elif property_type is not None:
            self.value_type = property_type
        else:
            self.value_type = type(default_value)
        self.default_value = default_value
        self.range = options_range
        self.possible_values = possible_values
        self.single_step = single_steep
        if self.value_type is list:
            assert default_value in options_range

    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}(name='{self.name}'," \
               f" user_name='{self.user_name}', " + \
               f"default_value={self.default_value}, type={self.value_type}, range={self.range}," \
               f"possible_values={self.possible_values})"


class AlgorithmDescribeBase:
    @classmethod
    def get_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def get_fields(cls) -> typing.List[AlgorithmProperty]:
        raise NotImplementedError()
