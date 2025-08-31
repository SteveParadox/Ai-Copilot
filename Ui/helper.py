# ui/helpers.py
import tkinter as tk

def set_hover(widget, enter_color="#dcdcdc", leave_color="#f0f0f0"):
    """
    Adds hover effect to a widget.
    """
    def on_enter(e):
        widget.config(bg=enter_color)

    def on_leave(e):
        widget.config(bg=leave_color)

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)

def update_status(label, text):
    """
    Updates a Tkinter Label with status text.
    """
    label.config(text=text)
    label.update_idletasks()

def copy_to_clipboard(root, text):
    """
    Copies text to system clipboard.
    """
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()  # now it stays on clipboard after app closes
