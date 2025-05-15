# Magic Object Finder 🔍

![Magic Object Finder](https://img.shields.io/badge/Magic%20Object%20Finder-Child%20Friendly-ff69b4)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![React](https://img.shields.io/badge/React-18.x-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C)

## Overview

Magic Object Finder is a delightful, child-friendly educational application that makes learning about the world fun and interactive! The app uses machine learning to identify objects in images and provides engaging, educational facts about what it finds.

## Features

- **Child-Friendly Design**: Baby pink background with animated snowflakes, Mickey Mouse silhouettes, candy canes, and a snow pile
- **Object Recognition**: Identifies 50+ different objects using a powerful machine learning model
- **Educational Content**: Provides fun, age-appropriate facts about recognized objects
- **Human Recognition**: Special detection for babies, boys, girls, and men with unique educational facts for each
- **Simple Interface**: Easy-to-use image upload functionality designed for children
- **Visual Feedback**: Confidence indicators and encouraging messages

## Tech Stack

- **Frontend**: React.js with custom CSS animations
- **Backend**: FastAPI with CORS support
- **ML Model**: EfficientNet B3 pre-trained on ImageNet
- **Image Processing**: PyTorch and PIL

## Setup Instructions

### Prerequisites
```bash
# Python 3.9+ required
python -m pip install fastapi uvicorn torch torchvision pillow numpy

# Node.js 14+ required
cd object-info-app
npm install
```

### Running the App

1. Start the ML Backend:
```bash
python fixed_backend.py
```

2. Start the React Frontend:
```bash
cd object-info-app
npm start
```

3. Open http://localhost:3001 in your browser

## Model Details

- **Architecture**: EfficientNet B3 pre-trained on ImageNet
- **Dataset**: Uses classes inspired by CIFAR-100 but leverages ImageNet pre-trained weights
- **Recognition**: 50+ different objects including animals, household items, people, and more
- **Processing**: Real-time analysis with confidence scoring
- **Face Detection**: Enhanced recognition of human faces
- **Training**: The model uses transfer learning with custom class mapping from ImageNet to our target classes

## Usage

1. Open the app in your browser
2. Click on the upload area or drag and drop an image
3. Wait a moment for the magic to happen!
4. See the prediction with fun facts and emojis
5. Click "Let's Try Another!" to test a different image

## Object Categories

The app can recognize a wide variety of objects, including:

### People 
- Baby 
- Boy
- Girl 
- Man 

### Animals 
- Bear 
- Butterfly 
- Cat 
- Dolphin 
- Elephant 
- Lion 
- Tiger 
- Bee 
- Beaver 

### Household Items 
- Bed 
- Chair 
- Clock 
- Keyboard 
- Lamp 

### Food & Containers 
- Apple 
- Bottle 
- Bowl 
- Cup 

### Nature 
- Cloud 
- Mountain 
- Rose 

### Vehicles 
- Bicycle 
- Bus 

##  Educational Content

Each object comes with child-friendly educational facts. 

## Project Structure

```
ObjectInfoML/
├── fixed_backend.py      # FastAPI backend with ML model
├── model.pth             # Trained model weights
├── README.md            # This documentation
├── object-info-app/     # React frontend
    ├── src/
    │   ├── App.js        # Main application component
    │   ├── App.css      # Main application styles
    │   ├── ImageUpload.js # Image upload component
    │   ├── ImageUpload.css # Image upload styles
    │   ├── ResultDisplay.js # Results display component
    │   ├── ResultDisplay.css # Results display styles
    │   └── index.js     # React entry point
    └── public/          # Static assets
```

##  Installation Requirements

### Backend
- Python 3.9+
- FastAPI
- Uvicorn
- PyTorch
- Torchvision
- Pillow
- NumPy

### Frontend
- Node.js 14+
- React 18+
- Modern web browser with JavaScript enabled

##  Acknowledgments

- ImageNet dataset
- PyTorch community
- React and FastAPI documentation
- All the children who inspire us to make learning fun!


