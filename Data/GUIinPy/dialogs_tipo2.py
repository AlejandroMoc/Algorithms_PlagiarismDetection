import tkinter as t
from tkinter import messagebox

def display_data(prompt, info):
    messagebox.showinfo(prompt, info)

def display_alert(prompt, info):
    messagebox.showwarning(prompt, info) 