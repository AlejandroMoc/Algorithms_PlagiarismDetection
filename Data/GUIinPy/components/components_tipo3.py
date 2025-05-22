import tkinter as tk
from tkinter import ttk, Checkbutton, Radiobutton, IntVar, Scale, HORIZONTAL, Text, Spinbox, Listbox, Canvas

# Define complex components here
def generar_control_deslizante(contenedor):
    control_deslizante = Scale(contenedor, from_=0, to=100, orient=HORIZONTAL, bg="#ffffff")
    control_deslizante.pack(pady=(10, 0))
    return control_deslizante

def generar_casilla_verificacion(contenedor):
    chk_var = IntVar()
    casilla_verif = Checkbutton(contenedor, text="Habilitar Función", variable=chk_var, bg="#ffffff")
    casilla_verif.pack(pady=(10, 0))
    return casilla_verif

def generar_radiobotones(contenedor):
    r_var = IntVar()
    rb1 = Radiobutton(contenedor, text="Opción 1", variable=r_var, value=1, bg="#ffffff")
    rb1.pack()
    rb2 = Radiobutton(contenedor, text="Opción 2", variable=r_var, value=2, bg="#ffffff")
    rb2.pack()
    return rb1, rb2

def generar_caja_texto(contenedor):
    caja_texto = Text(contenedor, height=5, width=30)
    caja_texto.pack(pady=(10, 0))
    return caja_texto

def generar_caja_numerica(contenedor):
    caja_numerica = Spinbox(contenedor, from_=0, to=10)
    caja_numerica.pack(pady=(10, 0))
    return caja_numerica

def generar_lista(contenedor):
    lista = Listbox(contenedor)
    lista.insert(1, "Elemento 1")
    lista.insert(2, "Elemento 2")
    lista.pack(pady=(10, 0))
    return lista

def generar_lienzo(contenedor):
    lienzo = Canvas(contenedor, width=100, height=100, bg="gris")
    lienzo.pack(pady=(10, 0))
    lienzo.create_oval(25, 25, 75, 75, fill="azul")
    return lienzo