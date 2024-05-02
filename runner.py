import time
import tkinter as tk

import numpy as np


from pynput.keyboard import Key, Controller


# start the root tk
def lazy_run():
    root = tk.Tk()
    root.title("Lazy Run")
    root.geometry("200x200")
    # overlay transparent window top but have the lines be visible
    # root.attributes("-alpha", 0.5)

    root.attributes("-topmost", True)
    root.attributes("-transparent", True)
    # root.attributes("-alpha", 0.0)


    canvas = tk.Canvas(root, width=200, height=200)
    canvas.pack()

    # demo arr
    arr = np.random.randint(0, 2, (1_000, 4), dtype=bool)

    for dir_set in arr:
        lazy_realtime_direction(root, canvas, dir_set)

    root.mainloop()
    # exit
    root.destroy()




# frame by frame in: [bool, bool, bool, bool] (up, down, left, right)
def lazy_realtime_direction(root, canvas, dirs1d: np.ndarray):
    keyboard = Controller()

    # clear the liens
    canvas.delete("all")
    # if the bool is true, then show the arrow, else remove it
    if dirs1d[0]:
        canvas.create_line(100, 100, 100, 50, arrow=tk.LAST)
        keyboard.press(Key.up)
        keyboard.release(Key.up)

    if dirs1d[1]:
        canvas.create_line(100, 100, 100, 150, arrow=tk.LAST)
        keyboard.press(Key.down)
        keyboard.release(Key.down)

    if dirs1d[2]:
        canvas.create_line(100, 100, 50, 100, arrow=tk.LAST)
        keyboard.press(Key.left)
        keyboard.release(Key.left)
    if dirs1d[3]:
        canvas.create_line(100, 100, 150, 100, arrow=tk.LAST)
        keyboard.press(Key.right)
        keyboard.release(Key.right)


    root.update()

if __name__ == '__main__':
    lazy_run()






# # input for realtime directional function: np array (bool) [[up, down, left, right], ...]
#
# # have an overlay of four arrows on the top right of the screen that is sticky and on top of everything
#
# # example arr:     directions = [[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]
# def realtime_directional(arr: np.ndarray, frame_rate: int = 5):
#     root = tk.Tk()
#     root.title("Realtime Directional")
#     root.geometry("200x200")
#
#     canvas = tk.Canvas(root, width=200, height=200)
#     canvas.pack()
#
#     for dir_set in arr:
#         # clear the liens
#         canvas.delete("all")
#         # if the bool is true, then show the arrow, else remove it
#         if dir_set[0]:
#             canvas.create_line(100, 100, 100, 50, arrow=tk.LAST)
#         if dir_set[1]:
#             canvas.create_line(100, 100, 100, 150, arrow=tk.LAST)
#         if dir_set[2]:
#             canvas.create_line(100, 100, 50, 100, arrow=tk.LAST)
#         if dir_set[3]:
#             canvas.create_line(100, 100, 150, 100, arrow=tk.LAST)
#
#         # pause for frame rate adj
#         pause = 1 / frame_rate
#         time.sleep(pause)
#         root.update()
#
#     root.mainloop()
#     # exit
#     root.destroy()
#
#
# if __name__ == '__main__':
#     # create a new thread that runs this function
#
#
#
#     # arr = np.array([[True, True, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True]])
#     # generate random from ^
#     arr = np.random.randint(0, 2, (100, 4), dtype=bool)
#     realtime_directional(arr)