python
import tkinter as tk
from tkinter import ttk
from gui_logic import handle_input
from components import create_slider, create_checkbox, create_radiobuttons, create_textbox, create_spinbox, create_listbox, create_canvas
from menus import create_menu
from dialogs import show_info, show_warning

def process_input():
    user_text = entry.get()
    feedback = handle_input(user_text)
    if user_text:
        show_info("Your Text", feedback)
    else:
        show_warning("Input Error", feedback)

root_window = tk.Tk()
root_window.title("Modern Python GUI")
root_window.geometry("500x300")
root_window.configure(bg="#f7f7f7")

main_frame = tk.Frame(root_window, bg="#ffffff", bd=10, relief=tk.RAISED)
main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

slider_widget = create_slider(main_frame)
checkbox_widget = create_checkbox(main_frame)
radiobuttons_widget = create_radiobuttons(main_frame)
textbox_widget = create_textbox(main_frame)
spinbox_widget = create_spinbox(main_frame)
listbox_widget = create_listbox(main_frame)
canvas_widget = create_canvas(main_frame)

menu_bar = create_menu(root_window)

entry = ttk.Entry(main_frame, font=("Arial", 12), width=40)
entry.pack(pady=(0, 20))
submit_button = ttk.Button(main_frame, text="Submit", command=process_input)
submit_button.pack(pady=(0, 10))

root_window.mainloop()