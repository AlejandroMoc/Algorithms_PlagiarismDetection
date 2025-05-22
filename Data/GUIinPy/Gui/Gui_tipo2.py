import tkinter as tk
from tkinter import ttk
from gui_logic import handle_input
from components import create_slider, create_checkbox, create_radiobuttons, create_textbox, create_spinbox, create_listbox, create_canvas
from menus import create_menu
from dialogs import show_info, show_warning

# Function to handle button click
def process_input():
    user_data = entry.get()
    message_output = handle_input(user_data)
    if user_data:
        show_info("Your Data", message_output)
    else:
        show_warning("Input Issue", message_output)

# Setup the main GUI window
root_window = tk.Tk()
root_window.title("Creative Python Interface")
root_window.geometry("500x300")
root_window.configure(bg="#f7f7f7")

main_frame = tk.Frame(root_window, bg="#ffffff", bd=10, relief=tk.RAISED)
main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# Adding components
slide_control = create_slider(main_frame)
check_box = create_checkbox(main_frame)
radio_buttons = create_radiobuttons(main_frame)
text_box = create_textbox(main_frame)
spin_box = create_spinbox(main_frame)
list_box = create_listbox(main_frame)
draw_canvas = create_canvas(main_frame)

# Menu
menu_bar = create_menu(root_window)

# Entry and button
entry = ttk.Entry(main_frame, font=("Arial", 12), width=40)
entry.pack(pady=(0, 20))
submit_button = ttk.Button(main_frame, text="Submit", command=process_input)
submit_button.pack(pady=(0, 10))

root_window.mainloop()