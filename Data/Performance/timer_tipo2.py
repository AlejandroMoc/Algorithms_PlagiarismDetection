import time
from contextlib import (
    contextmanager,
)  # helps in defining a block of setup and teardown logic


@contextmanager
def chronos(description="Chronometry"):
    """
    A context manager that measures the execution time of a code block.

    Args:
        description (str, optional): A description of the operation being timed. Defaults to 'Chronometry'.

    Yields:
        None

    Example:
        with chronos('Temporal analysis'):
            # some time-consuming operation
            time.sleep(2)

    This will print:
    Temporal analysis took 2.00 seconds
    """
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"{description} took {elapsed_time:.2f} seconds")


# example usage
if __name__ == "__main__":
    with chronos("Pilot study"):
        # simulate some time-consuming operation
        time.sleep(2)