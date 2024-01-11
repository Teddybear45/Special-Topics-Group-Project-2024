

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
from tkinter import ttk, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import datetime
import tkinter as tk
from ultralytics import YOLO
from predict import predict_image, predict_video


class VideoPlayer:
    def __init__(self, canvas, video_path, fps=5):
        self.canvas = canvas
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.is_playing = False
        self.frame_delay = 1.0/fps  # Set the frame delay in milliseconds

    def start(self):
        self.is_playing = True
        self.thread.start()

    def stop(self):
        self.is_playing = False
        self.cap.release()

    def update(self):
        while self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                image_path = self.save_temp_image(frame)
                self.show_frame(image_path)
                # Add a delay between frames (frame_delay is in milliseconds)
                time.sleep(self.frame_delay)
            else:
                # self.is_playing = False

                # Video has reached the end, rewind to the beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def save_temp_image(self, image):
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image, 0))
        return temp_path

    def show_frame(self, image_path):
        img = Image.open(image_path)

        # Get the current window width
        window_width = self.canvas.winfo_width()

        # Calculate the scaling factor to fit the image into the window width
        scale_factor = window_width / img.width

        # Resize the image
        img = img.resize((int(img.width * scale_factor), int(img.height * scale_factor)), Image.LANCZOS)

        img = ImageTk.PhotoImage(img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"), width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor='nw', image=img)
        self.canvas.image = img

class ImageVideoViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image/Video Viewer")
        self.root.geometry("800x600")

        self.file_path = tk.StringVar()
        self.folder_path = tk.StringVar()
        self.image_type = "image"
        self.selected_image_type = tk.StringVar(value="original")
        
        self.model = YOLO("/Users/nathansun/Documents/Special-Topics-Group-Project-2024/best.pt")
        self.fps = tk.IntVar(self.root, 5)
        self.r = tk.IntVar(self.root, 255)
        self.g = tk.IntVar(self.root, 100)
        self.b = tk.IntVar(self.root, 100)
        self.alpha = tk.IntVar(self.root, 100)
        self.rgba = tk.StringVar(value=", ".join(str(i) for i in [self.r.get(), self.g.get(), self.b.get(), self.alpha.get()]))

        # GUI Components
        tk.Label(self.root, text="Select FPS:").pack(pady=10)

        fps_entry = tk.Entry(self.root, textvariable=self.fps, state='disabled', width=10)
        fps_entry.pack(pady=5)

        tk.Button(self.root, text="Set FPS", command=self.set_fps).pack(pady=10)

        rgba_entry = tk.Entry(self.root, textvariable=self.rgba, state='disabled', width=50)
        rgba_entry.pack(pady=5)

        tk.Button(self.root, text="Set Overlay RGBA", command=self.set_rgba).pack(pady=10)

        tk.Label(self.root, text="Select File:").pack(pady=10)

        entry_folder = tk.Entry(self.root, textvariable=self.file_path, state='disabled', width=20)
        entry_folder.pack(pady=5)

        tk.Button(self.root, text="Browse", command=self.browse_file).pack(pady=10)

        tk.Label(self.root, text="Select Image/Video:").pack(pady=10)

        tk.Radiobutton(self.root, text="Original", variable=self.selected_image_type, value="original",
                       command=self.update_view).pack()
        tk.Radiobutton(self.root, text="Overlay", variable=self.selected_image_type, value="overlay",
                       command=self.update_view).pack()
        tk.Radiobutton(self.root, text="Edges", variable=self.selected_image_type, value="edges",
                       command=self.update_view).pack()

        tk.Button(self.root, text="Predict!", command=self.predict).pack(pady=20)

        # Image/Video Display with Scrollbar
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(expand=True, fill='both')

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(side='left', expand=True, fill='both')

        scrollbar = ttk.Scrollbar(self.canvas_frame, orient='vertical', command=self.canvas.yview)
        scrollbar.pack(side='right', fill='y')
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.temp_image_path = None  # Track the temporary image path

    def browse_file(self):
        chosen_file = filedialog.askopenfilename()
        self.file_path.set(chosen_file)

    def set_fps(self):
        chosen_fps = simpledialog.askinteger("FPS Entry", "Enter FPS")
        self.fps.set(chosen_fps)

    def set_rgba(self):
        chosen_r = simpledialog.askinteger("RGBA Entry", "Enter Red value (0 - 255)")
        self.r.set(chosen_r)

        chosen_g = simpledialog.askinteger("RGBA Entry", "Enter Green value (0 - 255)")
        self.g.set(chosen_g)

        chosen_b = simpledialog.askinteger("RGBA Entry", "Enter Blue value (0 - 255)")
        self.b.set(chosen_b)

        chosen_a = simpledialog.askinteger("RGBA Entry", "Enter Alpha value (0 - 255)")
        self.alpha.set(chosen_a)

        self.rgba.set(", ".join(str(i) for i in [self.r.get(), self.g.get(), self.b.get(), self.alpha.get()]))

    def predict(self):
        file_path = self.file_path.get()

        current_time = datetime.datetime.now()
        #TODO: definetely better way to do this lol
        run_name = str(current_time.year) + str(current_time.month) + str(current_time.day) + str(current_time.hour) + str(current_time.minute) + str(current_time.second)
        os.makedirs(os.path.join("./cache/runs", run_name))
        #TODO: make params changable

        if file_path.endswith(".jpg"):
            predict_image(self.model, file_path, os.path.join("./cache/runs", run_name), 30, False, (self.r.get(), self.g.get(), self.b.get(), self.alpha.get()))
            self.image_type = "image"
        elif file_path.endswith(".mp4"):
            predict_video(self.model, file_path, os.path.join("./cache/runs", run_name), 30, False, (self.r.get(), self.g.get(), self.b.get(), self.alpha.get()), self.fps.get())
            self.image_type = "video"
    
        self.folder_path.set(os.path.join("./cache/runs", run_name))
        self.update_view()

    def update_view(self):
        file_path = self.file_path.get()
        folder_path = self.folder_path.get()

        fps = self.fps.get()

        r = self.r.get()
        g = self.g.get()
        b = self.b.get()

        alpha = self.alpha.get()
        self.rgba = value=", ".join(str(i) for i in [self.r.get(), self.g.get(), self.b.get(), self.alpha.get()])

        if not folder_path:
            return

        image_type = self.image_type
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

            if hasattr(self, "video_player") and self.video_player.is_playing:
                self.video_player.stop()

            self.video_player = VideoPlayer(self.canvas, video_path, fps=fps)
            self.video_player.start()
            return

        self.show_image(image_path)

    """def reset_view(self):
        self.folder_path.set("")
        self.image_type.set("image")
        self.selected_image_type.set("original")
        self.canvas.delete("all")

        # Clean up the temporary image file
        if self.temp_image_path and os.path.exists(self.temp_image_path):
            os.remove(self.temp_image_path)
            self.temp_image_path = None"""

    def show_image(self, image_path):
        self.canvas.delete("all")
        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"), width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor='nw', image=img)
        self.canvas.image = img

    def save_temp_image(self, image):
        temp_path = os.path.join("cache", "temp_image.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return temp_path

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageVideoViewer(root)
    root.mainloop()



