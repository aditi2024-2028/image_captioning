"""Desktop GUI for generating image captions with an E0-R checkpoint."""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk

from caption import CaptionGenerator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(BASE_DIR, "outputs", "e0r", "ckpt_epoch05.pt")
PREVIEW_SIZE = (680, 420)


class CaptionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Image Captioning")
        self.root.geometry("760x720")
        self.root.minsize(600, 560)

        self.checkpoint_path = tk.StringVar(value=DEFAULT_CHECKPOINT)
        self.status = tk.StringVar(value="Choose an image to get started.")
        self.selected_image = None
        self.preview = None
        self.generator = None
        self.generator_checkpoint = None
        self.busy = False

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)

        ttk.Label(container, text="Checkpoint").grid(row=0, column=0, sticky="w")
        checkpoint_row = ttk.Frame(container)
        checkpoint_row.grid(row=1, column=0, sticky="ew", pady=(4, 12))
        checkpoint_row.columnconfigure(0, weight=1)
        self.checkpoint_entry = ttk.Entry(
            checkpoint_row, textvariable=self.checkpoint_path
        )
        self.checkpoint_entry.grid(row=0, column=0, sticky="ew")
        self.checkpoint_button = ttk.Button(
            checkpoint_row, text="Browse...", command=self._choose_checkpoint
        )
        self.checkpoint_button.grid(row=0, column=1, padx=(8, 0))

        self.image_button = ttk.Button(
            container, text="Choose image...", command=self._choose_image
        )
        self.image_button.grid(row=2, column=0, sticky="w")

        self.image_label = ttk.Label(
            container, text="No image selected", anchor="center"
        )
        self.image_label.grid(row=3, column=0, sticky="nsew", pady=12)
        container.rowconfigure(3, weight=1)

        ttk.Label(container, text="Caption", font=("", 12, "bold")).grid(
            row=4, column=0, sticky="w", pady=(4, 4)
        )
        self.caption_label = ttk.Label(
            container,
            text="Your caption will appear here.",
            anchor="nw",
            justify="left",
            wraplength=700,
        )
        self.caption_label.grid(row=5, column=0, sticky="ew", pady=(0, 12))

        self.generate_button = ttk.Button(
            container,
            text="Generate caption",
            command=self._generate_caption,
            state="disabled",
        )
        self.generate_button.grid(row=6, column=0, sticky="w")
        ttk.Label(container, textvariable=self.status).grid(
            row=7, column=0, sticky="w", pady=(10, 0)
        )

    def _choose_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=(("PyTorch checkpoints", "*.pt *.pth"), ("All files", "*.*")),
            initialdir=os.path.dirname(self.checkpoint_path.get()) or BASE_DIR,
        )
        if path:
            self.checkpoint_path.set(path)
            self.generator = None
            self.generator_checkpoint = None

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail(PREVIEW_SIZE)
                self.preview = ImageTk.PhotoImage(image)
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open image", str(error))
            return

        self.selected_image = path
        self.image_label.configure(image=self.preview, text="")
        self.caption_label.configure(text="Your caption will appear here.")
        self.generate_button.configure(state="normal")
        self.status.set(os.path.basename(path))

    def _generate_caption(self) -> None:
        checkpoint = self.checkpoint_path.get().strip()
        if not checkpoint or not os.path.isfile(checkpoint):
            messagebox.showerror(
                "Checkpoint not found", "Select a valid model checkpoint first."
            )
            return
        if not self.selected_image or self.busy:
            return

        self.busy = True
        self._set_controls_enabled(False)
        self.status.set("Generating caption... The first run loads the model.")
        self.caption_label.configure(text="")

        image_path = self.selected_image
        device = "cuda" if torch.cuda.is_available() else "cpu"
        threading.Thread(
            target=self._run_caption,
            args=(image_path, checkpoint, device),
            daemon=True,
        ).start()

    def _run_caption(self, image_path: str, checkpoint: str, device: str) -> None:
        try:
            if self.generator is None or self.generator_checkpoint != checkpoint:
                self.generator = CaptionGenerator(checkpoint, device)
                self.generator_checkpoint = checkpoint
            caption = self.generator.caption_image(image_path)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            self.root.after(0, self._show_error, error)
            return

        self.root.after(0, self._show_caption, caption)

    def _show_caption(self, caption: str) -> None:
        self.caption_label.configure(text=caption)
        self.status.set("Caption generated.")
        self.busy = False
        self._set_controls_enabled(True)

    def _show_error(self, error: Exception) -> None:
        self.status.set("Caption generation failed.")
        self.busy = False
        self._set_controls_enabled(True)
        messagebox.showerror("Caption generation failed", str(error))

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.image_button.configure(state=state)
        self.checkpoint_button.configure(state=state)
        self.checkpoint_entry.configure(state=state)
        self.generate_button.configure(
            state="normal" if enabled and self.selected_image else "disabled"
        )


def main() -> None:
    root = tk.Tk()
    CaptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
