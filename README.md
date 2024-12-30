# Enhance Lipsynced Videos with Superresolution

## Overview
This project enhances lipsynced videos by applying superresolution to subframes where resolution is lower than the original input frames. The enhancements ensure that only the required regions (subframes) are improved, while preserving the entire video and its audio.

### Key Features
- Extracts frames from input and lipsynced videos.
- Detects faces and applies superresolution to those regions only.
- Maintains the original audio from the lipsynced video.
- Generates a high-quality enhanced video with the same audio.

## Prerequisites
### Python Version
- Python 3.x

### Required Libraries
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- MoviePy (for video/audio integration)

## Installation

### Clone the Repository
```bash
git clone https://github.com/TencentARC/GFPGAN.git
cd GFPGAN
pip install -r requirements.txt
```

### Download the Pre-trained Models
```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth -O experiments/pretrained_models/GFPGANv1.4.pth
```

### Install MoviePy for Audio-Video Combination
```bash
pip install moviepy
```

## Workflow

### 1. Input Videos and Audio
- **Videos**: Input video (e.g., lipsynced video) downloaded from platforms like Pexels.
- **Audio**: Audio extracted from video or downloaded from sources like Pixabay.

### 2. Lipsynced Video Generation
- Lipsynced video is created using tools like Voza to match audio with facial movements.

### 3. Superresolution Enhancement
- For subframes (faces), superresolution is applied using GFPGAN.
- Only the necessary subframes are enhanced, preserving the overall resolution of the rest of the frame.

### Example Command
```bash
python x.py --superres GFPGAN -iv input.mp4 -ia input.mp3 -o output.mp4
```

### 4. Audio Handling
- The final enhanced video retains the original audio from the lipsynced video.

## Project Structure
```
├── GFPGAN/                    # Superresolution model
│   ├── requirements.txt
│   ├── pretrained_models/
├── CodeFormer/                # Alternative Superresolution model
│   ├── requirements.txt
│   ├── pretrained_models/
├── input_frames/              # Extracted frames from input video
├── lipsynced_frames/          # Extracted frames from lipsynced video
├── enhanced_frames/           # Enhanced frames with superresolution
├── output_directory/          # Directory for final enhanced video and plots
└── x.py                       # Main script for enhancement process
```

## Notes
- Ensure both input video and audio paths are correctly specified.
- GFPGAN is used for superresolution enhancement in this project.
