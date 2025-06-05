import time

def chrono_wrapped_func(ma_func):
    def chrono(*args, **kwargs):
        """
        Decorates a function to print its execution time.
        """
        message_top = "\nInitiating {}".format(ma_func.__name__)
        print(message_top)
        print("-" * len(message_top))
        t0 = time.time()

        result = ma_func(*args, **kwargs)

        message_bot = "\nDone in {:.2f} minutes.".format((time.time() - t0)/60)
        print(message_bot)
        print('-' * len(message_bot))
        return result
    return chrono