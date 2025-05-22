import tkinter as tk
from tkinter import ttk, Checkbutton, Radiobutton, IntVar, Scale, HORIZONTAL, Text, Spinbox, Listbox, Canvas

# Define wacky components here
def generate_gauge(window):
    gauge = Scale(window, from_=0, to=100, orient=HORIZONTAL, bg="#ffffff")
    gauge.pack(pady=(10, 0))
    return gauge

def design_toggle(window):
    tgl_var = IntVar()
    tgl = Checkbutton(window, text="Enable Feature", variable=tgl_var, bg="#ffffff")
    tgl.pack(pady=(10, 0))
    return tgl

def setup_radio(window):
    r_var = IntVar()
    rb1 = Radiobutton(window, text="Option 1", variable=r_var, value=1, bg="#ffffff")
    rb1.pack()
    rb2 = Radiobutton(window, text="Option 2", variable=r_var, value=2, bg="#ffffff")
    rb2.pack()
    return rb1, rb2

def construct_text_area(window):
    txt_area = Text(window, height=5, width=30)
    txt_area.pack(pady=(10, 0))
    return txt_area

def assemble_selector(window):
    selector = Spinbox(window, from_=0, to=10)
    selector.pack(pady=(10, 0))
    return selector

def build_menu(window):
    menu = Listbox(window)
    menu.insert(1, "Item 1")
    menu.insert(2, "Item 2")
    menu.pack(pady=(10, 0))
    return menu

def compose_art(window):
    art = Canvas(window, width=100, height=100, bg="grey")
    art.pack(pady=(10, 0))
    art.create_oval(25, 25, 75, 75, fill="blue")
    return art 