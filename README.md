A Multimodal Approach to Upper Limb Stroke Rehabilitation:
EMG Signal Analysis and Video-Based Motion Tracking
--------------------------------------------------------------
## Authors
* **Maria Fidalgo** 
* **Mariana Morais** - [mariana-m0rais](https://github.com/mariana-m0rais)

*Master in Biomedical Engineering | University of Coimbra*

## Context
This project was developed in the academic year 2024/2025 as part of the curriculum for the Master's degree in Biomedical Engineering at the University of Coimbra, specifically for the course of Clinical Computing and Telehealth Systems. 


## Overview
This project implements a multimodal system for analyzing upper limb movement in the context of stroke rehabilitation. It combines electromyography (EMG) data acquisition from a BITalino device with real-time video capture and pose estimation using MediaPipe. The system allows for synchronized collection of physiological and visual data, enabling detailed analysis of muscle activation patterns and limb movement. The output includes both a processed EMG signal and the real-time count and analysis of arm movements performed by the user.


## Requirements
This project was developed and tested with Python 3.10. 

It uses the BITalino (r)evolution Python API to interface with the BITalino board via Bluetooth.
To install and configure it correctly, please follow the official installation instructions provided in the repository:
🔗 https://github.com/BITalinoWorld/revolution-python-api/blob/master/README.md
Make sure to follow all prerequisites and steps outlined there for your operating system.

The additional necessary packages can be installed with:
** pip install numpy matplotlib scipy opencv-python mediapipe **


## System Setup & Results

### 1. Hardware & Experimental Setup
The system utilizes a BITalino (r)evolution board configured for surface electromyography (sEMG) acquisition. Electrodes are placed over the *biceps brachii* muscle to capture signal bursts during elbow flexion exercises.

<img src="https://github.com/user-attachments/assets/b6844bbc-cf8a-40a8-ad1a-0888fa555ed5" width="50%" alt="Hardware & Experimental Setup" />

### 2. Signal Processing & Activation Detection
The raw sEMG signal is filtered (band-pass) and processed using the Hilbert Transform to extract the smooth amplitude envelope. Voluntary muscle contractions are automatically isolated based on a dynamic threshold.

<img src="https://github.com/user-attachments/assets/f5bca3a0-3fd7-4991-9706-77cc415397a6" width="40%" alt="Signal Processing & Activation Detection" />

### 3. Video-Based Motion Tracking
Simultaneously, a separate thread manages the computer vision pipeline using MediaPipe Pose. The algorithm computes real-time anatomical joint angles to track exercise execution, count repetitions, and monitor postural range of motion.

<img src="https://github.com/user-attachments/assets/d93e957f-b6fe-4952-9bb9-b96137601e8e" width="50%" alt="Video-Based Motion Tracking" />

## Outputs
Real-time display of:
   - Repetitions detected based on elbow angle.
   - Annotated webcam feed with pose landmarks.

Plots:
   - Raw EMG signal
   - Filtered signal with activation highlights

Console output with:
   - Detected activation summary (start time, end time, duration, amplitude)

Video file video_output.avi
Processed data file emg_processed_data.npz
