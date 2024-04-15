import os
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import datetime
from ultralytics import YOLO

from application import pred_move_video
from predict import predict_image, predict_video_as_frames

class VideoPlayer:
    def __init__(self, video_path, fps=5):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.window = tk.Toplevel()
        self.canvas = tk.Canvas(self.window)
        self.canvas.pack()
        self.is_playing = False
        self.frame_delay = 1.0 / fps  # Set the frame delay in milliseconds

    def start(self):
        self.is_playing = True
        self.update()

    def stop(self):
        self.is_playing = False
        self.cap.release()
        self.window.destroy()

    def restart_in_same_window(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            image_path = self.save_temp_image(frame)
            self.show_frame(image_path)
            self.window.after(int(self.frame_delay * 1000), self.update)
        else:
            # self.stop() # TODO
            self.restart_in_same_window()


    def save_temp_image(self, image):
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image, 0))
        return temp_path

    def show_frame(self, image_path):
        img = Image.open(image_path)

        # Get the size of the canvas
        # make the canvas size proportional to the image size but have it be a fixed 800px width

        canvas_width = 800
        canvas_height = int(img.height * (canvas_width / img.width))

        # Resize the image to fit the canvas
        img = img.resize((canvas_width, canvas_height))

        img = ImageTk.PhotoImage(img)

        # Clear previous content on the canvas
        self.canvas.delete("all")

        # Display the resized image on the canvas
        # have the width and heigh of the canvas be proportional to the width and height of the image; but have it be a fixed 800px width
        # self.canvas.config(scrollregion=self.canvas.bbox("all"), width=img.width(), height=img.height())
        self.canvas.config(scrollregion=self.canvas.bbox("all"), width=canvas_width, height=canvas_height)
        self.canvas.create_image(0, 0, anchor='nw', image=img)
        self.canvas.image = img


        # img = ImageTk.PhotoImage(img)
        # self.canvas.config(width=img.width(), height=img.height())
        # self.canvas.create_image(0, 0, anchor='nw', image=img)
        # self.canvas.image = img



class ImageVideoViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image/Video Viewer")
        self.root.geometry("800x600")

        self.file_path = tk.StringVar()
        self.folder_path = tk.StringVar()
        self.image_type = "image"
        self.selected_image_type = tk.StringVar(value="original")

        self.model = YOLO("/Users/nathansun/Documents/Special-Topics-Group-Project-2024.nosync/best.pt")
        self.fps = tk.IntVar(self.root, 5)
        self.r = tk.IntVar(self.root, 255)
        self.g = tk.IntVar(self.root, 100)
        self.b = tk.IntVar(self.root, 100)
        self.alpha = tk.IntVar(self.root, 100)
        self.rgba = tk.StringVar(value=", ".join(str(i) for i in [self.r.get(), self.g.get(), self.b.get(), self.alpha.get()]))

        self.temp_image_path = None

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
        tk.Radiobutton(self.root, text="Path", variable=self.selected_image_type, value="path", command=self.update_view).pack()

        tk.Button(self.root, text="Predict!", command=self.predict).pack(pady=20)

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
        run_name = str(current_time.year) + str(current_time.month) + str(current_time.day) + str(
            current_time.hour) + str(current_time.minute) + str(current_time.second)

        cache_run_path = os.path.join("./cache/runs", run_name)
        os.makedirs(cache_run_path)

        if file_path.endswith(".jpg"):
            predict_image(self.model, file_path, os.path.join("./cache/runs", run_name), 30, False,
                          (self.r.get(), self.g.get(), self.b.get(), self.alpha.get()))
            self.image_type = "image"
        elif file_path.endswith(".mp4"):
            predict_video_as_frames(self.model, file_path, os.path.join("./cache/runs", run_name), 30, False,
                          (self.r.get(), self.g.get(), self.b.get(), self.alpha.get()), self.fps.get())

            angles = pred_move_video(cache_run_path, -1, 15, (self.r.get(), self.g.get(), self.b.get(), self.alpha.get()), self.fps.get())

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

            elif selected_type == "edges":
                original_video = cv2.VideoCapture(f"{folder_path}/original.mp4")
                binary_mask_video_of_edges = cv2.VideoCapture(f"{folder_path}/edges.mp4")

                # for every frame in the orginal video, overlay the binary mask with the mask color
                # save the frames to a temp video file
                # set the video path to the temp video file
                video_path = binary_mask_video_of_edges # TODO

            elif selected_type == "path":
                video_path = f"{folder_path}/path_overlay.mp4"




            # if hasattr(self, "video_player") and self.video_player.is_playing:
            #     self.video_player.stop()

            # have it loop, actually: TODO


            self.video_player = VideoPlayer(video_path, fps=fps)
            self.video_player.start()
            return

        self.show_image(image_path)

    def show_image(self, image_path):
        window = tk.Toplevel()
        canvas = tk.Canvas(window)
        canvas.pack()

        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)
        canvas.config(scrollregion=canvas.bbox("all"), width=img.width(), height=img.height())
        canvas.create_image(0, 0, anchor='nw', image=img)
        canvas.image = img

    def save_temp_image(self, image):
        temp_path = os.path.join("cache", "temp_image.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, 0))
        return temp_path

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageVideoViewer(root)
    root.mainloop()
