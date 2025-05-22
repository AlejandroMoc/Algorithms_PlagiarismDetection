from functools import wraps


def cache_result(func):
    """
    A decorator that caches the result of the function, avoiding repeated computations.

    This decorator is useful for expensive function calls. It will store the results of the function
    for a given set of arguments and return the cached results when the function is called again with
    the same arguments.

    Args:
        func (callable): The function to be memoized.

    Returns:
        callable: A wrapper function that implements memoization.

    Example:
        @cache_result
        def fib(n):
            if n < 2:
                return n:
            return fib(n-1) + fib(n-2)

        # first call will compute the result
        print(fib(10))
        # subsequent calls with the same argument will return the cached result
        print(fib(10))
    """
    cache = {}

    @wraps(func)  # copies/retains the metadata from the original function
    def caching_wrapper(*args, **kwargs):
        # create a key that uniquely identifies the function call
        key = str(args) + str(kwargs)

        # if this function call is not cached, compute and cache the result
        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    return caching_wrapper


# example usage
if __name__ == "__main__":

    @cache_result
    def expensive_op(a, b):
        import time

        time.sleep(2)  # simulate an expensive operation
        return a + b

    # first call will take about 2 seconds
    print(expensive_op(2, 3))

    # second call with the same arguments will be instant
    print(expensive_op(2, 3))
