python
import tkinter as tk
from tkinter import messagebox

def mostrar_informacion(titulo, mensaje):
    messagebox.showinfo(titulo, mensaje)

def mostrar_advertencia(titulo, mensaje):
    messagebox.showwarning(titulo, mensaje)