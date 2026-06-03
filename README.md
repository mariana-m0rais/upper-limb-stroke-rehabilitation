A Multimodal Approach to Upper Limb Stroke Rehabilitation:
EMG Signal Analysis and Video-Based Motion Tracking
--------------------------------------------------------------
By Maria Fidalgo (2021235050) and Mariana Morais (2021236380)


Overview:
This project implements a multimodal system for analyzing upper limb movement in the context of stroke rehabilitation. It combines electromyography (EMG) data acquisition from a BITalino device with real-time video capture and pose estimation using MediaPipe. The system allows for synchronized collection of physiological and visual data, enabling detailed analysis of muscle activation patterns and limb movement. The output includes both a processed EMG signal and the real-time count and analysis of arm movements performed by the user.


Requirements:
This project was developed and tested with Python 3.10. 

It uses the BITalino (r)evolution Python API to interface with the BITalino board via Bluetooth.
To install and configure it correctly, please follow the official installation instructions provided in the repository:
🔗 https://github.com/BITalinoWorld/revolution-python-api/blob/master/README.md
Make sure to follow all prerequisites and steps outlined there for your operating system.

The necessary packages can be installed with:
pip install numpy matplotlib scipy opencv-python mediapipe


Outputs:
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