# ui/window.py
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from helper import set_hover, update_status, copy_to_clipboard

class InterviewCopilot(tk.Tk):
    """
    Main Tkinter GUI for Interview Copilot
    """
    def __init__(self):
        super().__init__()
        self.title("Interview Copilot")
        self.geometry("700x500")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")

        self.create_widgets()

    def create_widgets(self):
        # Status Label
        self.status_label = ttk.Label(self, text="Ready", background="#f0f0f0")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Text Area for Transcript
        self.transcript_area = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, height=20, width=80
        )
        self.transcript_area.pack(padx=10, pady=10)
        self.transcript_area.config(state=tk.DISABLED)

        # Control Buttons
        button_frame = tk.Frame(self, bg="#f0f0f0")
        button_frame.pack(pady=5)

        self.start_btn = tk.Button(button_frame, text="Start", command=self.start_capture)
        self.stop_btn = tk.Button(button_frame, text="Stop", command=self.stop_capture)
        self.copy_btn = tk.Button(button_frame, text="Copy Transcript", command=self.copy_transcript)

        for btn in [self.start_btn, self.stop_btn, self.copy_btn]:
            btn.pack(side=tk.LEFT, padx=5)
            set_hover(btn)

    def start_capture(self):
        update_status(self.status_label, "Listening...")
        # Hook in AudioCapture start (to be passed from main pipeline)
        # self.audio_capture.start()

    def stop_capture(self):
        update_status(self.status_label, "Stopped")
        # Hook in AudioCapture stop
        # self.audio_capture.stop()

    def append_transcript(self, text):
        """
        Appends text to transcript area safely
        """
        self.transcript_area.config(state=tk.NORMAL)
        self.transcript_area.insert(tk.END, text + "\n")
        self.transcript_area.yview(tk.END)
        self.transcript_area.config(state=tk.DISABLED)

    def copy_transcript(self):
        text = self.transcript_area.get("1.0", tk.END)
        copy_to_clipboard(self, text)
        messagebox.showinfo("Copied", "Transcript copied to clipboard!")
