from package.show import display
from package.write import plot


def scribble(pencil_case, secret_key):
    return plot(pencil_case, secret_key)


def reveal(pencil_case, secret_key, clipboard=None):
    return display(pencil_case, secret_key, clipboard=None)
