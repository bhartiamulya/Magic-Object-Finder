import React, { useState, useRef } from 'react';
import './ImageUpload.css';

const ImageUpload = ({ setError, setIsLoading, setPredictions }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mode, setMode] = useState('upload'); // 'upload' or 'camera'
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const handleImageChange = (event) => {
    console.log('File input change detected');
    const file = event.target.files[0];
    if (file) {
      console.log('File selected:', file.name);
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      console.log('No file selected');
    }
  };

  const triggerFileInput = () => {
    console.log('Triggering file input click');
    document.getElementById('file-input').click();
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
    } catch (err) {
      setError("Oops! I couldn't access your camera. 📸 Can you make sure it's allowed?");
      setMode('upload');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    canvas.toBlob((blob) => {
      setSelectedImage(blob);
      setPreviewUrl(canvas.toDataURL('image/jpeg'));
      stopCamera();
    }, 'image/jpeg');
  };

  const handleSubmit = async () => {
    if (!selectedImage) {
      setError("Please select or capture an image first! 📸");
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result = await response.json();
      setPredictions(result.predictions);
    } catch (error) {
      setError("Oops! Sorry I don't recognise it. Let's try again! 🔄");
    } finally {
      setIsLoading(false);
    }
  };

  React.useEffect(() => {
    return () => {
      stopCamera();
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  React.useEffect(() => {
    if (mode === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
  }, [mode]);

  return (
    <div className="upload-container">
      <div className="mode-buttons">
        <button
          className={`mode-button ${mode === 'upload' ? 'active' : ''}`}
          onClick={() => {
            setMode('upload');
            // Add a small delay to ensure the mode is set before triggering the file input
            setTimeout(triggerFileInput, 100);
          }}
        >
          📷 Upload Picture
        </button>
        <button
          className={`mode-button ${mode === 'camera' ? 'active' : ''}`}
          onClick={() => setMode('camera')}
        >
          📱 Use Camera
        </button>
      </div>

      {mode === 'upload' ? (
        <div className="upload-area">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="file-input"
            id="file-input"
          />
          <div className="file-label" onClick={triggerFileInput}>
            {previewUrl ? (
              <img src={previewUrl} alt="Preview" className="preview-image" />
            ) : (
              <div className="upload-placeholder">
                <span className="upload-icon">📁</span>
                <p>Click here to choose a picture!</p>
                <p className="upload-hint">or drag and drop one here</p>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="camera-area">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className={previewUrl ? 'hidden' : 'camera-preview'}
          />
          {previewUrl && (
            <img src={previewUrl} alt="Captured" className="preview-image" />
          )}
          {!previewUrl && (
            <button className="capture-button" onClick={captureImage}>
              📸 Take Picture!
            </button>
          )}
        </div>
      )}

      {previewUrl && (
        <div className="button-container">
          <button 
            className="submit-button"
            onClick={handleSubmit}
          >
            🔍 What's in this picture?
          </button>
          <button 
            className="reset-button"
            onClick={() => {
              setSelectedImage(null);
              setPreviewUrl(null);
              if (mode === 'camera') {
                startCamera();
              }
            }}
          >
            🔄 Try Another Picture
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
