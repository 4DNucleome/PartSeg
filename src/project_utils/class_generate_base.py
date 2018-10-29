import collections

class BaseReadonlyClass:
    def asdict(self) -> collections.OrderedDict:
        pass