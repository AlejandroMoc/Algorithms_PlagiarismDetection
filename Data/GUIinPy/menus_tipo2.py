import tkinter as tk
from tkinter import Menu

def generate_ui(root):
    menubar = Menu(root)
    menu_a = Menu(menubar, tearoff=0)
    menu_a.add_command(label="Open", command=lambda: None)
    menu_a.add_command(label="Save", command=lambda: None)
    menu_a.add_separator()
    menu_a.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=menu_a)
    root.config(menu=menubar)
    return menubar 