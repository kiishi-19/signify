import cv2
import numpy as np
import subprocess as sp
import time
import csv
import os
from datetime import datetime
import mediapipe as mp # type: ignore
import os
from dotenv import load_dotenv # type: ignore

load_dotenv()

## Prepare camera process and hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

width, height = 3840, 2160
fps = 30
frame_size = width * height * 3 // 2

# Commands for the cameras
command1 = ['libcamera-vid', '-n', '--width', str(width), '--height', str(height), '--framerate', str(fps), '--codec', 'yuv420', '-o', '-', '--timeout', '0', '--inline', '--camera', '0']
command2 = ['libcamera-vid', '-n', '--width', str(width), '--height', str(height), '--framerate', str(fps), '--codec', 'yuv420', '-o', '-', '--timeout', '0', '--inline', '--camera', '1']

process1 = sp.Popen(command1, stdout=sp.PIPE, bufsize=width * height * 3 // 2)
process2 = sp.Popen(command2, stdout=sp.PIPE, bufsize=width * height * 3 // 2)

# Prepare the CSV files to store the landmark data
base_directory = os.getenv("BASE_DIRECTORY")
if base_directory is None:
    raise ValueError("BASE_DIRECTORY is not set in the .env file")
base_filename = "hello"

index_file_path = os.path.join(base_directory, "index.txt")
os.makedirs(base_directory, exist_ok=True)

if os.path.exists(index_file_path):
    with open(index_file_path, "r") as index_file:
        current_number = int(index_file.read().strip())
else:
    current_number = 0

filename1 = os.path.join(base_directory, f"{base_filename}_camera1_{current_number:04d}.csv")
filename2 = os.path.join(base_directory, f"{base_filename}_camera2_{current_number:04d}.csv")

current_number += 1

with open(index_file_path, "w") as index_file:
    index_file.write(str(current_number))

# Countdown before starting recording
countdown_duration = 5
print(f"Recording will start in {countdown_duration} seconds...")
for i in range(countdown_duration, 0, -1):
    print(i)
    time.sleep(1)
print("Recording started!")

# Loop to process frames and collect landmark data
try:
    while True:
        stdout_data1 = process1.stdout.read(frame_size)
        stdout_data2 = process2.stdout.read(frame_size)

        if len(stdout_data1) == frame_size and len(stdout_data2) == frame_size:
            raw_frame1 = np.frombuffer(stdout_data1, dtype=np.uint8)
            raw_frame2 = np.frombuffer(stdout_data2, dtype=np.uint8)

            yuv_frame1 = raw_frame1.reshape((height + height // 2, width))
            bgr_frame1 = cv2.cvtColor(yuv_frame1, cv2.COLOR_YUV2BGR_I420)

            yuv_frame2 = raw_frame2.reshape((height + height // 2, width))
            bgr_frame2 = cv2.cvtColor(yuv_frame2, cv2.COLOR_YUV2BGR_I420)

            results1 = hands.process(cv2.cvtColor(bgr_frame1, cv2.COLOR_BGR2RGB))
            results2 = hands.process(cv2.cvtColor(bgr_frame2, cv2.COLOR_BGR2RGB))

            with open(filename1, 'a', newline='') as csv_file1:
                csv_writer1 = csv.writer(csv_file1)
                if results1.multi_hand_landmarks:
                    for hand_landmarks in results1.multi_hand_landmarks:
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        csv_writer1.writerow(landmarks)
                else:
                    csv_writer1.writerow([0] * 63)

            with open(filename2, 'a', newline='') as csv_file2:
                csv_writer2 = csv.writer(csv_file2)
                if results2.multi_hand_landmarks:
                    for hand_landmarks in results2.multi_hand_landmarks:
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        csv_writer2.writerow(landmarks)
                else:
                    csv_writer2.writerow([0] * 63)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    process1.terminate()
    process1.wait()
    process2.terminate()
    process2.wait()