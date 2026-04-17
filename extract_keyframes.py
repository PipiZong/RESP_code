import cv2
import os
from pathlib import Path
from PIL import Image
import numpy as np
# extract ten frames from a video
frames_dir = 'keyframes_physgame'
os.makedirs(frames_dir, exist_ok=True)

# function to extract keyframes from a video using ffmpeg
def extract_keyframes(video_path, frames_dir):
    video_name = Path(video_path).stem
    video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)

    command = f"ffmpeg -i {video_path} -vf \"select='eq(pict_type\\,I)'\" -vsync vfr -q:v 1 {video_frames_dir}/%06d.jpg"
    # command = f"ffmpeg -i {video_path} -vf fps=2 -q:v 1 {video_frames_dir}/%06d.jpg"
    # extract keyframes using ffmpeg and record their timestamp
    # command = f"ffmpeg -i {video_path} -vf \"select='eq(pict_type\\,I)'\" -vsync vfr -frame_pts true {video_frames_dir}/%04d.jpg"
    os.system(command)
    print(f"Extracted keyframes from {video_path} to {video_frames_dir}")

def extract_frames(video_path, frames_dir, num_frames=None):
    video_name = Path(video_path).stem

    video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames is None:
        num_frames = frame_count
    frame_indices = np.linspace(0, frame_count - 1, num_frames).astype(int)

    extracted_count = 0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if i in frame_indices:
            frame_path = os.path.join(video_frames_dir, f"{extracted_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path} to {video_frames_dir}")

video_folder = 'physgame_videos'
for video in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video)
    extract_keyframes(video_path, frames_dir)
    # extract_frames(video_path, frames_dir, num_frames=50)
