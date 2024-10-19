from time import sleep, time
from gpiozero import Servo, LED
from gpiozero.pins.pigpio import PiGPIOFactory

import pygame
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

import cv2
import threading

jaw_pin = 17
neck_pin = 27
led_pin = 22

factory = PiGPIOFactory()
#jaw_factory = PiGPIOFactory()

jaw = Servo(jaw_pin, pin_factory=factory, initial_value=-1)
neck = Servo(neck_pin, pin_factory=factory, initial_value=0)
led = LED(led_pin, pin_factory=factory)

laugh_filepath = "/home/jared/Downloads/laugh.mp3"
howl_filepath = "/home/jared/Downloads/howl.mp3"
doodle_filepath = "/home/jared/Downloads/doodling.mp3"
iownyou_filepath = "/home/jared/Downloads/iownyou.mp3"
shortfilm_filepath = "/home/jared/Downloads/shortfilm.mp3"


face_cascade = cv2.CascadeClassifier('/home/jared/Downloads/haarcascade_frontalface_default.xml')


pygame.mixer.init()

def get_audio_amplitudes(audio_file_path, chunk_size_ms=50):
    audio = AudioSegment.from_file(audio_file_path)
    chunks = make_chunks(audio, chunk_size_ms)
    
    amplitudes = [chunk.dBFS for chunk in chunks]
    
    return np.array(amplitudes)

def amplitude_to_jaw_position(amplitude, min_db, max_db, jaw_range):
    
    if amplitude == -np.inf:
        amplitude = min_db - 10
        
    if max_db == min_db:
        return -1
    
    normalized_amplitude = (amplitude - min_db) / (max_db - min_db)    
    jaw_position = normalized_amplitude * jaw_range - 1    
    return jaw_position

def set_jaw_position(value):
    if value < -1:
        value = -1
    elif value > 0:
        value = 0
    jaw.value = value

def set_neck_position(value):
    if value < -1:
        value = -1
    elif value > .5:
        value = .5
    neck.value = value
    
def sync_track(filepath, jaw_range):
    pygame.mixer.music.load(filepath)
    
    amplitudes = get_audio_amplitudes(filepath)
    
    min_db = np.min(amplitudes[amplitudes != -np.inf])
    max_db = np.max(amplitudes)
    
    pygame.mixer.music.play()    
    
    for amplitude in amplitudes:
        jaw_position = amplitude_to_jaw_position(amplitude, min_db, max_db, jaw_range)
        print(jaw_position)
        set_jaw_position(jaw_position)
        sleep(0.05)
        
    set_jaw_position(-1)




def simulate_laugh():

    sync_track(laugh_filepath, .25)    


def howl():
    

    sync_track(howl_filepath, 1)    

            
def doodle():

    sync_track(doodle_filepath, .5)    
    


def iownyou():
    
    sync_track(iownyou_filepath, .5)    
    
    
def shortfilm():
    
    sync_track(shortfilm_filepath, .75)    
    

def track_nearest_object():
    cap = cv2.VideoCapture(0)
    screen_center_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
    
    previous_object_detected = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize = (30,30))
        
        if len(objects) > 0:
            objects = sorted(objects, key=lambda x: x[2] * x[3], reverse=True)
            x,y,w,h = objects[0]
            object_center_x = x + w // 2
            
            offset = (object_center_x - screen_center_x) / screen_center_x
            
            set_neck_position(offset)
            
            if not previous_object_detected:
                previous_object_detected = True
                #sound_thread = threading.Thread(target=iownyou, args=())
                #sound_thread.start()
        else:
            previous_object_detected = False
            
        #cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()



led.on()


track_nearest_object()


led.off()



# jaw.close()
# neck.close()
# led.close()

print("done")
