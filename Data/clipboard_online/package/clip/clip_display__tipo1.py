from package.show import display
from package.write import plot

def write_text(text_name, key):
    return plot(text_name, key)

def show_text(text_name, key, clipboard=None):
    return display(text_name, key, clipboard=None)