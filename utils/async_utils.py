import asyncio
from typing import Any, Callable, Coroutine


def run_async(coro: Coroutine) -> Any:
    """Run an async coroutine synchronously. Safe for LangGraph sync nodes."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


def make_sync(async_fn: Callable[..., Coroutine]) -> Callable:
    """Wrap an async function to be callable synchronously."""
    def wrapper(*args, **kwargs):
        return run_async(async_fn(*args, **kwargs))
    return wrapper
