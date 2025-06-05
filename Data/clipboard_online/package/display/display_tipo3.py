from package.show import display
from package.write import plot






def show(snippet_name, password, clipboard=None):
    return display(snippet_name, password, clipboard=None)


def write(snippet_name, password):
    return plot(snippet_name, password)
