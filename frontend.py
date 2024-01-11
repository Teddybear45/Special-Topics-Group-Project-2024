import tkinter as tk


# Premise
# gui using Tkinter that will have a button for selecting the folder for the results. Below that, there's a radio button that asks whether it's image or video format

# for image, the directory chosen will have this structure:
# original.jpg (or jpeg)
# overlay.jpg (or jpeg)
# edges.jpg (or jpeg)

# for this, show the original image. then, on the right side have a radio check that will allow the user to select which image to show (original, overlay, edges)
# for the edges, we need to do some calculations wherein the white (255) pixels are the edges. we can do this by taking the difference between the original and the overlay,
# \ and then thresholding it at 0. this will give us a binary image where the edges are white and the rest is black. we can then display this image if that's the selection

# for video, the directory chosen will have this structure:
# original.mp4
# overlay.mp4

# for this, show the original video. then, on the right side have a radio check that will allow the user to select which video to show (original, overlay)



# finally, there should be a reset/back button to choose a new directory

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import os
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk

class ImageVideoViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image/Video Viewer")
        self.root.geometry("800x600")

        self.folder_path = tk.StringVar()
        self.image_type = tk.StringVar(value="image")
        self.selected_image_type = tk.StringVar(value="original")

        # GUI Components
        tk.Label(self.root, text="Select Folder:").pack(pady=10)
        entry_folder = tk.Entry(self.root, textvariable=self.folder_path, state='disabled', width=50)
        entry_folder.pack(pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_folder).pack(pady=10)

        tk.Label(self.root, text="Select Type:").pack(pady=10)
        tk.Radiobutton(self.root, text="Image", variable=self.image_type, value="image", command=self.update_view).pack()
        tk.Radiobutton(self.root, text="Video", variable=self.image_type, value="video", command=self.update_view).pack()

        tk.Label(self.root, text="Select Image/Video:").pack(pady=10)
        tk.Radiobutton(self.root, text="Original", variable=self.selected_image_type, value="original",
                       command=self.update_view).pack()
        tk.Radiobutton(self.root, text="Overlay", variable=self.selected_image_type, value="overlay",
                       command=self.update_view).pack()
        tk.Radiobutton(self.root, text="Edges", variable=self.selected_image_type, value="edges",
                       command=self.update_view).pack()

        tk.Button(self.root, text="Reset/Back", command=self.reset_view).pack(pady=20)

        # Image/Video Display with Scrollbar
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(expand=True, fill='both')

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(side='left', expand=True, fill='both')

        scrollbar = ttk.Scrollbar(self.canvas_frame, orient='vertical', command=self.canvas.yview)
        scrollbar.pack(side='right', fill='y')
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.temp_image_path = None  # Track the temporary image path

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path.set(folder_selected)
            self.update_view()

    def update_view(self):
        folder_path = self.folder_path.get()

        if not folder_path:
            return

        image_type = self.image_type.get()
        selected_type = self.selected_image_type.get()

        if image_type == "image":
            image_path = f"{folder_path}/original.jpg"
            if selected_type == "overlay":
                image_path = f"{folder_path}/overlay.jpg"
            elif selected_type == "edges":
                original_image = cv2.imread(f"{folder_path}/original.jpg", cv2.IMREAD_GRAYSCALE)
                binary_mask_image_of_edges = cv2.imread(f"{folder_path}/edges.jpg", cv2.IMREAD_GRAYSCALE)
                overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                overlay_image[binary_mask_image_of_edges == 255] = [255, 0, 0]
                if self.temp_image_path and os.path.exists(self.temp_image_path):
                    image_path = self.temp_image_path
                else:
                    image_path = self.save_temp_image(overlay_image)
                    self.temp_image_path = image_path

        elif image_type == "video":
            video_path = f"{folder_path}/original.mp4"
            if selected_type == "overlay":
                video_path = f"{folder_path}/overlay.mp4"
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                if self.temp_image_path and os.path.exists(self.temp_image_path):
                    image_path = self.temp_image_path
                else:
                    image_path = self.save_temp_image(frame)
                    self.temp_image_path = image_path
                self.show_image(image_path)
                return
            else:
                self.canvas.delete("all")
                tk.Label(self.root, text="Unable to load video").pack()
                return

        self.show_image(image_path)

    def reset_view(self):
        self.folder_path.set("")
        self.image_type.set("image")
        self.selected_image_type.set("original")
        self.canvas.delete("all")

        # Clean up the temporary image file
        if self.temp_image_path and os.path.exists(self.temp_image_path):
            os.remove(self.temp_image_path)
            self.temp_image_path = None

    def show_image(self, image_path):
        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"), width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor='nw', image=img)
        self.canvas.image = img

    def save_temp_image(self, image):
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return temp_path

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageVideoViewer(root)
    # root.protocol("WM_DELETE_WINDOW", root.destroy) # Handle the window close event
    root.mainloop()


