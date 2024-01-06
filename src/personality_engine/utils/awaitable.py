import asyncio
from functools import wraps, partial
from typing import TypeVar, Callable, ParamSpec, Coroutine, Any

T = TypeVar('T')
P = ParamSpec('P')


def awaitable(func: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T]]:
	@wraps(func)
	async def run(*args: P.args, **kwargs: P.kwargs):
		loop = asyncio.get_running_loop()
		fn = partial(func, *args, **kwargs)
		return await loop.run_in_executor(None, fn)

	return run
