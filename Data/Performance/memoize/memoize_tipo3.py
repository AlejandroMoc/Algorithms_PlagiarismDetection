from functools import wraps


def cache_result(function):
    cache = {}

    @wraps(function)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = function(*args, **kwargs)
        return cache[key]

    return wrapper


if __name__ == "__main__":

    @cache_result
    def compute_sum(a, b):
        import time
        time.sleep(2)
        return a + b

    print(compute_sum(2, 3))
    print(compute_sum(2, 3))