# -*- coding: utf-8 -*-
"""
A Multimodal Approach to Upper Limb Stroke Rehabilitation: 
EMG Signal Analysis and Video-Based Motion Tracking
-------------------------------------------------------------
Maria Fidalgo (2021235050) and Mariana Morais (2021236380)
"""

import time
import bluetooth
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import uniform_filter1d
from bitalino import BITalino
import cv2
import mediapipe as mp
from threading import Thread

# Bluetooth Device Search
def list_devices():
    print("Searching for Bluetooth devices...")
    devices = bluetooth.discover_devices(duration=8, lookup_names=True)
    for i, (addr, name) in enumerate(devices):
        print(f"{i+1}. {name} - {addr}")
    return devices

devices = list_devices()
if not devices:
    print("No devices found.")
    exit()
choice = int(input("\nSelect the device number: ")) - 1
macAddress = devices[choice][0]

# Connect to BITalino
device = BITalino(macAddress)
device.battery(30)
print(f"BITalino version: {device.version()}")
device.start(1000, [0])  # Sampling rate = 1000 Hz, channel 0

# Initialize webcam and MediaPipe
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening webcam.")
    device.close()
    exit()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video_output.avi', fourcc, 30, (width, height))
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Data Storage
raw_emg = []
timestamps = []
frame_timestamps, angles = [], []
reps = 0
stage = None
asktime = int(input("\nSelect the running time (seconds): "))
print("The aquisition will begin.")
start = time.time()

# Thread for EMG Acquisition 
def emg_thread():
    while (time.time() - start) < asktime:
        samples = device.read(25)
        now = time.time() - start
        for sample in samples:
            raw_emg.append(sample[-1])
            timestamps.append(now)

# Thread for Video + Pose

def calculate_angle(a, b, c): 
    # Angle calculation function 
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def video_thread():
    global reps, stage
    while (time.time() - start) < asktime:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time() - start
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lms = results.pose_landmarks.landmark
            shoulder = [lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [lms[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     lms[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [lms[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     lms[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)
            angles.append(angle)
            frame_timestamps.append(now)
            cv2.putText(frame, str(int(angle)),
                        tuple(np.multiply(elbow, [width, height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if angle > 170:
                stage = "up"
            if angle < 40 and stage == "up":
                stage = "down"
                reps += 1
                print(f"Repetitions: {reps}")
        else:
            angles.append(np.nan)
            frame_timestamps.append(now)
        cv2.putText(frame, f"Reps: {reps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Live Webcam', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Run Threads
t1 = Thread(target=emg_thread)
t2 = Thread(target=video_thread)
t1.start()
t2.start()
t1.join()
t2.join()

# Stop Acquisition
device.stop()
device.close()
cap.release()
out.release()
cv2.destroyAllWindows()
print("Acquisition finished.")

# Convert Raw Signal
def convert_to_millivolts(signal, adc_resolution=10, vcc=3.3, gain=1000):
    adc_max = 2**adc_resolution - 1
    voltage = (signal / adc_max) * vcc
    mv_signal = (voltage / gain) * 1000
    return mv_signal

raw_emg_array = np.array(raw_emg)
emg_mv_raw = convert_to_millivolts(raw_emg_array)

# Plot Raw EMG Signal
plt.figure(figsize=(12, 4))
plt.plot(timestamps, emg_mv_raw, color='orange', label='Raw EMG Signal (mV)')
plt.title('Raw EMG Signal (No Filtering)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid()
plt.show()

# Filter Signal
def highpass_filter(signal, cutoff=20, fs=1000, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

def lowpass_filter(signal, cutoff=450, fs=1000, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def apply_filters(signal_mV):
    filtered_hp = highpass_filter(signal_mV)
    filtered_lp = lowpass_filter(filtered_hp)
    return filtered_lp

emg_filtered = apply_filters(emg_mv_raw)

# Hilbert Transform & Envelope Extraction
analytic_signal = hilbert(emg_filtered)
envelope = np.abs(analytic_signal)
envelope_smooth = uniform_filter1d(envelope, size=20)

threshold = np.mean(envelope_smooth) + 2 * np.std(envelope_smooth)
# print(f"Envelope-based threshold: {threshold:.2f} mV")

# Detect Single Continuous Activation
activations = []
activation_detected = False
start_idx = None
end_idx = None
min_duration_samples = 60
gap_tolerance = 200

i = 0
while i < len(envelope_smooth):
    if envelope_smooth[i] > threshold:
        if not activation_detected:
            start_idx = i
            activation_detected = True
        below_threshold_counter = 0
    elif activation_detected:
        below_threshold_counter += 1
        if below_threshold_counter >= gap_tolerance:
            end_idx = i - gap_tolerance
            activation_detected = False
            if (end_idx - start_idx) >= min_duration_samples:
                duration = timestamps[end_idx] - timestamps[start_idx]
                amplitude = np.max(envelope_smooth[start_idx:end_idx]) - np.min(envelope_smooth[start_idx:end_idx])
                activations.append({
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx],
                    'duration': duration,
                    'amplitude': amplitude,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
    i += 1

# Plot Envelope with Activation Highlighted 
plt.figure(figsize=(12, 4))
plt.plot(timestamps, envelope_smooth, label='Hilbert Envelope (smoothed)', color='purple')
for act in activations:
    act_time = timestamps[act['start_idx']:act['end_idx']]
    act_env = envelope_smooth[act['start_idx']:act['end_idx']]
    plt.plot(act_time, act_env, color='green', linewidth=2.5, label='Detected Activation')
plt.title("Hilbert Envelope of EMG Signal with Detected Activation")
plt.xlabel("Time (s)")
plt.ylabel("Envelope Amplitude (mV)")
plt.legend()
plt.grid()
plt.show()

# Print Activation Summary 
print("\n--- Activation Summary ---")
for idx, act in enumerate(activations, 1):
    print(f"Activation {idx}:")
    print(f"  Start Time: {act['start_time']:.3f} s")
    print(f"  End Time: {act['end_time']:.3f} s")
    print(f"  Duration: {act['duration']:.3f} s")
    print(f"  Amplitude: {act['amplitude']:.2f} mV")
print(f"\nTotal Activations Detected: {len(activations)}")

# Save Processed Data 
np.savez("emg_processed_data.npz",
         timestamps=timestamps,
         envelope=envelope_smooth,
         raw_signal=emg_mv_raw,
         filtered_signal=emg_filtered)
print("Data saved at emg_processed_data.npz ")
