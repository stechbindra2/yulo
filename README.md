# PCB Defect Detection System

A system for detecting defects in printed circuit boards using YOLOv8 object detection.

## Features

- Detect common PCB manufacturing defects
- Interactive UI for image upload and result visualization
- Available as both desktop and web applications
- Dark/light mode support

## Defect Types Detected

- Missing Hole
- Mouse Bite
- Open Circuit
- Short
- Spur
- Spurious Copper

## Setup Instructions

### Prerequisites

- Python 3.7 or newer
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd yulo
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure the trained model file `best.pt` is in the project directory.

## Running the Desktop Application

```bash
python yolo_gui.py
```

### Desktop Application Usage:

1. Click "Select Image" to choose a PCB image for analysis
2. Click "Detect Defects" to analyze the image
3. View results in the results panel
4. Use "Save Results" to save the annotated image
5. Use "Clear" to reset and analyze another image

## Running the Web Application

```bash
python app.py
```

Then open a web browser and navigate to `http://127.0.0.1:5000/`

### Web Application Usage:

1. Upload a PCB image by clicking "Browse Files" or using drag and drop
2. Click "Detect Defects" to analyze the image
3. View the detection results with highlighted defects
4. Use the "Download" button to save the annotated image
5. Click "New Detection" to analyze another image

## Project Structure

- `yolo_gui.py`: Desktop application using Tkinter
- `app.py`: Flask web application
- `templates/`: HTML templates for web application
- `static/`: CSS, JavaScript, and static assets
- `best.pt`: Trained YOLOv8 model for PCB defect detection
- `requirements.txt`: Required Python packages

## Dependencies

- ultralytics (YOLOv8)
- opencv-python
- pillow
- numpy
- flask (for web app)
- tkinter (for desktop app)
