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
        self.selected_images = []
        self.preview = None
        self.generator = None
        self.generator_checkpoint = None
        self.busy = False
        self.captions = []
        self.result_thumbnail_images = {}
        self.result_caption_labels = {}

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

        image_row = ttk.Frame(container)
        image_row.grid(row=2, column=0, sticky="ew")
        self.image_button = ttk.Button(
            image_row, text="Add one image...", command=self._choose_images
        )
        self.image_button.pack(side="left")
        self.clear_button = ttk.Button(
            image_row, text="Clear", command=self._clear_images, state="disabled"
        )
        self.clear_button.pack(side="left", padx=(8, 0))

        images_row = ttk.Frame(container)
        images_row.grid(row=3, column=0, sticky="nsew", pady=12)
        images_row.columnconfigure(1, weight=1)
        container.rowconfigure(3, weight=1)

        self.image_list = tk.Listbox(
            images_row, height=10, exportselection=False, selectmode="browse"
        )
        self.image_list.grid(row=0, column=0, rowspan=2, sticky="nsw")
        list_scroll = ttk.Scrollbar(
            images_row, orient="vertical", command=self.image_list.yview
        )
        list_scroll.grid(row=0, column=1, rowspan=2, sticky="ns")
        self.image_list.configure(yscrollcommand=list_scroll.set)
        self.image_list.bind("<<ListboxSelect>>", self._preview_selected_image)

        self.image_label = ttk.Label(
            images_row, text="No images selected", anchor="center"
        )
        self.image_label.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        images_row.rowconfigure(0, weight=1)
        images_row.columnconfigure(2, weight=1)

        ttk.Label(container, text="Captions", font=("", 12, "bold")).grid(
            row=4, column=0, sticky="w", pady=(4, 4)
        )
        caption_row = ttk.Frame(container)
        caption_row.grid(row=5, column=0, sticky="nsew", pady=(0, 12))
        caption_row.columnconfigure(0, weight=1)
        container.rowconfigure(5, weight=1)

        caption_row.rowconfigure(0, weight=1)
        self.results_canvas = tk.Canvas(caption_row, height=260)
        self.results_canvas.grid(row=0, column=0, sticky="nsew")
        caption_scroll = ttk.Scrollbar(
            caption_row, orient="vertical", command=self.results_canvas.yview
        )
        caption_scroll.grid(row=0, column=1, sticky="ns")
        self.results_canvas.configure(yscrollcommand=caption_scroll.set)
        self.results_frame = ttk.Frame(self.results_canvas)
        self.results_canvas_window = self.results_canvas.create_window(
            (0, 0), window=self.results_frame, anchor="nw"
        )
        self.results_frame.bind(
            "<Configure>",
            lambda event: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            ),
        )
        self.results_canvas.bind(
            "<Configure>",
            lambda event: self.results_canvas.itemconfigure(
                self.results_canvas_window, width=event.width
            ),
        )

        actions_row = ttk.Frame(container)
        actions_row.grid(row=6, column=0, sticky="w")
        self.generate_button = ttk.Button(
            actions_row,
            text="Generate selected caption",
            command=self._generate_caption,
            state="disabled",
        )
        self.generate_button.pack(side="left")
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

    def _rebuild_results(self) -> None:
        for child in self.results_frame.winfo_children():
            child.destroy()
        self.result_thumbnail_images.clear()
        self.result_caption_labels.clear()

        for index, path in enumerate(self.selected_images):
            row = ttk.Frame(
                self.results_frame,
                padding=(6, 6),
                relief="solid",
                borderwidth=1,
            )
            row.pack(fill="x", expand=True, pady=3)
            row.columnconfigure(1, weight=1)

            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((170, 110))
                    thumbnail = ImageTk.PhotoImage(image)
            except (OSError, ValueError):
                thumbnail = None

            image_label = ttk.Label(row)
            if thumbnail is not None:
                self.result_thumbnail_images[index] = thumbnail
                image_label.configure(image=thumbnail)
            else:
                image_label.configure(text="Image unavailable")
            image_label.grid(row=0, column=0, padx=(0, 10), sticky="nw")

            text_frame = ttk.Frame(row)
            text_frame.grid(row=0, column=1, sticky="nsew")
            ttk.Label(
                text_frame,
                text=os.path.basename(path),
                font=("", 9, "bold"),
                anchor="w",
            ).pack(fill="x")
            caption_label = ttk.Label(
                text_frame,
                text=self.captions[index] if self.captions else "Not generated",
                anchor="nw",
                justify="left",
                wraplength=430,
            )
            caption_label.pack(fill="x", pady=(4, 0))
            self.result_caption_labels[index] = caption_label

    def _choose_images(self) -> None:
        path = filedialog.askopenfilename(
            title="Add one image",
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        try:
            with Image.open(path) as source:
                source.verify()
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open image", str(error))
            return

        self.selected_images.append(path)
        self.captions.append("Not generated")
        index = len(self.selected_images) - 1
        self.image_list.insert(tk.END, os.path.basename(path))
        self.image_list.selection_clear(0, tk.END)
        self.image_list.selection_set(index)
        self.image_list.activate(index)
        self.image_list.see(index)
        self._preview_selected_image()

        self._rebuild_results()

        self.clear_button.configure(state="normal")
        self.generate_button.configure(state="normal")
        self.status.set(
            f"Added image {index + 1}. Select it and generate its caption."
        )

    def _clear_images(self) -> None:
        if self.busy:
            return
        self.selected_images = []
        self.captions = []
        self.preview = None
        self.image_list.delete(0, tk.END)
        self._rebuild_results()
        self.image_label.configure(image="", text="No images selected")
        self.clear_button.configure(state="disabled")
        self.generate_button.configure(state="disabled")
        self.status.set("Choose images to get started.")

    def _preview_selected_image(self, event=None) -> None:
        selection = self.image_list.curselection()
        if not selection:
            return
        path = self.selected_images[selection[0]]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail(PREVIEW_SIZE)
                self.preview = ImageTk.PhotoImage(image)
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open image", str(error))
            return

        self.image_label.configure(image=self.preview, text="")

    def _generate_caption(self) -> None:
        checkpoint = self.checkpoint_path.get().strip()
        if not checkpoint or not os.path.isfile(checkpoint):
            messagebox.showerror(
                "Checkpoint not found", "Select a valid model checkpoint first."
            )
            return
        selection = self.image_list.curselection()
        if not selection or self.busy:
            return

        index = selection[0]
        self.busy = True
        self._set_controls_enabled(False)
        self.captions[index] = "Generating..."
        self.result_caption_labels[index].configure(text="Generating...")
        self.status.set(
            f"Generating caption for image {index + 1}... The first run loads the model."
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        threading.Thread(
            target=self._run_caption,
            args=(self.selected_images[index], index, checkpoint, device),
            daemon=True,
        ).start()

    def _run_caption(
        self, image_path: str, index: int, checkpoint: str, device: str
    ) -> None:
        try:
            if self.generator is None or self.generator_checkpoint != checkpoint:
                self.generator = CaptionGenerator(checkpoint, device)
                self.generator_checkpoint = checkpoint
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            self.root.after(0, self._show_error, error, index)
            return

        try:
            caption = self.generator.caption_image(image_path)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            self.root.after(0, self._show_caption_error, index, error)
            return
        self.root.after(0, self._show_caption_result, index, caption)

    def _show_caption_result(self, index: int, caption: str) -> None:
        self.captions[index] = caption
        self.result_caption_labels[index].configure(text=caption)
        self.busy = False
        self._set_controls_enabled(True)
        self.status.set(
            f"Caption added to this session. {len(self.selected_images)} image(s) in results."
        )

    def _show_caption_error(self, index: int, error: Exception) -> None:
        self.captions[index] = "Generation failed"
        self.result_caption_labels[index].configure(text="Generation failed")
        self.status.set(f"Caption generation failed for image {index + 1}.")
        self.busy = False
        self._set_controls_enabled(True)
        messagebox.showerror("Caption generation failed", str(error))

    def _show_error(self, error: Exception, index: int) -> None:
        self.captions[index] = "Generation failed"
        self.result_caption_labels[index].configure(text="Generation failed")
        self.status.set("Caption generation failed.")
        self.busy = False
        self._set_controls_enabled(True)
        messagebox.showerror("Caption generation failed", str(error))

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.image_button.configure(state=state)
        self.clear_button.configure(
            state="normal" if enabled and self.selected_images else "disabled"
        )
        self.checkpoint_button.configure(state=state)
        self.checkpoint_entry.configure(state=state)
        self.generate_button.configure(
            state="normal"
            if enabled and self.image_list.curselection()
            else "disabled"
        )


def main() -> None:
    root = tk.Tk()
    CaptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
