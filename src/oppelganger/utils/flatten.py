import functools
import operator
from typing import TypeVar, List

T = TypeVar('T')


def flatten(some_list: List[List[T]]) -> List[T]:
	return functools.reduce(operator.iconcat, some_list, [])
