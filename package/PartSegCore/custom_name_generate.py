import random
import string
from typing import Any, Dict, Set


def custom_name_generate(prohibited_names: Set[str], dict_names: Dict[str, Any]) -> str:
    letters_set = string.ascii_letters + string.digits
    for _ in range(1000):
        rand_name = "custom_" + "".join(random.choice(letters_set) for _ in range(10))  # nosec  # noqa: S311 #NOSONAR
        if rand_name not in prohibited_names and rand_name not in dict_names:
            return rand_name
    raise RuntimeError("Cannot generate proper names")  # pragma: no cover
