from pynput import mouse, keyboard
import numpy as np
import pyautogui
import time
import win32api
import pydirectinput
import win32con

Start_detection = False
Listen = True
width = 0
interval = 0.01
screen_size = np.array([win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)])
screen_center=np.array(screen_size,dtype=int)//2
destination = screen_center

def get_S_L():
    global Start_detection
    global Listen
    return Start_detection, Listen


# if the esc is clicked, return True
def listen_key(key):
    if key == keyboard.Key.home:
        global Start_detection
        global Listen
        Listen = False
        Start_detection = False
        print("Stop listening")


# if the right mouse is clicked, return True
def listen_mouse(x, y, button, pressed):
    global Start_detection
    if button == mouse.Button.right:
        if pressed:
            Start_detection = not Start_detection
            print("Start detection: ", Start_detection)

def speed_func(x, speed, smooth):
    global width
    return max(x/smooth/speed,width/3)

def Move_Mouse(args):
    global screen_size,screen_center
    #while Listen:
    global destination, width, interval
    if Start_detection:
        pos=np.array(pydirectinput.position(),dtype=int)
        mouse_vector = (destination - pos)*2/3
        norm = np.linalg.norm(mouse_vector)
        #if destination not in region
        if norm <= 2 or (destination[0]==screen_center[0] and destination[1]==screen_center[1]):
            return

        # normalize mouse_vector
        # normalized_vector = mouse_vector * 1.0 / norm
        # des = normalized_vector * speed_func(norm, args.mouse_speed, args.smooth)

        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,int(mouse_vector[0]/3), int(mouse_vector[1]/3))



# redirect the mouse closer to the nearest box center
def Mouse_redirection(boxes, args, tpf):
    global destination, width, interval, screen_size, screen_center
    if boxes.shape[0] == 0:
        destination=screen_center
        return
    interval = tpf
    pos = np.array(pydirectinput.position())

    # Get the center of the boxes
    boxes_center = (
        (boxes[:, :2] + boxes[:, 2:]) / 2
    )
    boxes_center[:, 1] = (
        boxes[:, 1] * 0.5 + boxes[:, 3] * 0.5
    )
    # Map the box from the image coordinate to the screen coordinate
    screen_center = screen_size / 2
    start_point = screen_center- screen_size[1] * args.crop_size / 2
    start_point=list(map(int, start_point))
    boxes_center[:, 0] = boxes_center[:, 0] + start_point[0]
    boxes_center[:, 1] = boxes_center[:, 1] + start_point[1]

    # Find the nearest box center
    dis = np.linalg.norm(boxes_center - pos, axis=-1)
    min_index = np.argmin(dis)
    width = boxes[min_index, 2] - boxes[min_index, 0]
    destination = boxes_center[np.argmin(dis)].astype(int)
    # print(destination)

    # mouse_instance.position = tuple(boxes_center[np.argmin(dis)])
