python
import time
from contextlib import (
    contextmanager,
)  # helps in defining a block of setup and teardown logic


@contextmanager
def tiempo(descripcion="Ejecución"):
    """
    A context manager that measures the execution time of a code block.

    Args:
        description (str, optional): A description of the operation being timed. Defaults to 'Execution'.

    Yields:
        None

    Example:
        with timer('Database query'):
            # some time-consuming operation
            time.sleep(2)

    This will print:
    Database query took 2.00 seconds
    """
    inicio = time.time()
    yield
    transcurrido = time.time() - inicio
    print(f"{descripcion} took {transcurrido:.2f} seconds")


# example usage
if __name__ == "__main__":
    with tiempo("Operación de ejemplo"):
        # simulate some time-consuming operation
        time.sleep(2)