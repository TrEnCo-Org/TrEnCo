from functools import cmp_to_key
from typing import Callable
from typing import TypeVar, Generic

U = TypeVar('U')

class Comparator(Generic[U]):    
    __func: Callable[[U, U], bool]

    def __init__(
        self,
        func: Callable[[U, U], bool]
    ) -> None:
        self.__func = func
        def cmp(x: U, y: U) -> int:
            if self.__func(x, y):
                return -1
            elif self.__func(y, x):
                return 1
            else:
                return 0
        self.__key = cmp_to_key(cmp)
    
    def __call__(self, x: U, y: U) -> bool:
        return self.__func(x, y)

    def key(self, x: U):
        return self.__key(x)