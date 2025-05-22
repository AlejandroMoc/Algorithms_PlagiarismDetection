import tkinter as tk
from tkinter import Menu

def generar_menu(ventana):
    barra_menu = Menu(ventana)
    menu_archivo = Menu(barra_menu, tearoff=0)
    menu_archivo.add_command(label="Abrir", command=lambda: None)
    menu_archivo.add_command(label="Guardar", command=lambda: None)
    menu_archivo.add_separator()
    menu_archivo.add_command(label="Salir", command=ventana.quit)
    barra_menu.add_cascade(label="Archivo", menu=menu_archivo)
    ventana.config(menu=barra_menu)
    return barra_menu