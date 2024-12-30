# -*- coding: utf-8 -*-
"""Enhance Lipsynced Videos with Superresolution Script"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, AudioFileClip
import argparse

def combine_video_audio(input_video_path, input_audio_path, output_video_path):
    try:
        # Load video and audio clips
        video = VideoFileClip(input_video_path)
        audio = AudioFileClip(input_audio_path)

        # Combine video and audio
        final_video = video.set_audio(audio)

        # Save the final video
        final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
        print(f"Combined video saved as {output_video_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_paths = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    cap.release()
    return frame_paths

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def apply_superresolution(frame, region):
    x, y, w, h = region
    cropped = frame[y:y+h, x:x+w]

    # Placeholder for superresolution logic
    enhanced = cv2.resize(cropped, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    frame[y:y+h, x:x+w] = cv2.resize(enhanced, (w, h))  # Resize back to original region size
    return frame

def enhance_lipsynced_video(input_video, lipsynced_video, input_audio, output_video):
    input_frames_dir = "input_frames"
    lipsynced_frames_dir = "lipsynced_frames"
    enhanced_frames_dir = "enhanced_frames"

    os.makedirs(enhanced_frames_dir, exist_ok=True)

    input_frames = extract_frames(input_video, input_frames_dir)
    lipsynced_frames = extract_frames(lipsynced_video, lipsynced_frames_dir)

    for input_frame_path, lipsynced_frame_path in zip(input_frames, lipsynced_frames):
        input_frame = cv2.imread(input_frame_path)
        lipsynced_frame = cv2.imread(lipsynced_frame_path)

        lipsynced_faces = detect_faces(lipsynced_frame)

        enhanced_frame = lipsynced_frame.copy()

        for (lx, ly, lw, lh) in lipsynced_faces:
            enhanced_frame = apply_superresolution(enhanced_frame, (lx, ly, lw, lh))

        frame_name = os.path.basename(lipsynced_frame_path)
        cv2.imwrite(os.path.join(enhanced_frames_dir, frame_name), enhanced_frame)

    # Create enhanced video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("temp_video.mp4", fourcc, 30, (int(cv2.VideoCapture(input_video).get(3)), int(cv2.VideoCapture(input_video).get(4))))

    for frame_file in sorted(os.listdir(enhanced_frames_dir)):
        frame = cv2.imread(os.path.join(enhanced_frames_dir, frame_file))
        out.write(frame)
    out.release()

    # Combine video with audio
    combine_video_audio("temp_video.mp4", input_audio, output_video)
    os.remove("temp_video.mp4")
    print(f"Enhanced video with audio saved at {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance Lipsynced Videos with Superresolution")
    parser.add_argument("--input_video", "-iv", type=str, required=True, help="Path to the input video")
    parser.add_argument("--lipsynced_video", "-lv", type=str, required=True, help="Path to the lipsynced video")
    parser.add_argument("--input_audio", "-ia", type=str, required=True, help="Path to the input audio")
    parser.add_argument("--output_video", "-o", type=str, required=True, help="Path to the output enhanced video")
    args = parser.parse_args()

    enhance_lipsynced_video(args.input_video, args.lipsynced_video, args.input_audio, args.output_video)
