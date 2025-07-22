import { useRef, useEffect, useState } from 'react';
import './App.css'; // Ensure you have the necessary styles
import * as faceapi from 'face-api.js';

function Attend() {
  const videoRef = useRef();
  const canvasRef = useRef();
  const [isFaceDetected, setIsFaceDetected] = useState(false);

  // Load models and start video
  useEffect(() => {
    const loadModels = async () => {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('/models'),
        faceapi.nets.ageGenderNet.loadFromUri('/models'),
      ]);
      startVideo();
    };

    loadModels();
  }, []);

  const startVideo = () => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((currentStream) => {
        videoRef.current.srcObject = currentStream;
        detectFace();
      })
      .catch((err) => {
        console.error(err);
      });
  };

  const detectFace = async () => {
    const displaySize = { width: 640, height: 480 };
    faceapi.matchDimensions(canvasRef.current, displaySize);

    setInterval(async () => {
      const detections = await faceapi.detectAllFaces(videoRef.current,
        new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions();

      if (detections.length > 0) {
        setIsFaceDetected(true);
      } else {
        setIsFaceDetected(false);
      }

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      canvasRef.current.getContext('2d').clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
      faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections);
      faceapi.draw.drawFaceExpressions(canvasRef.current, resizedDetections);
    }, 100);
  };

  const registerFace = () => {
    if (isFaceDetected) {
      console.log("Face registered!"); // Logic to register the face
      // Here you would typically call your face recognition API or logic
    } else {
      alert("No face detected. Please ensure your face is visible.");
    }
  };

  return (
    <div className="attend">
      <h3>Face Recognition Attendance</h3>
      <div className="video-container">
        <video ref={videoRef} autoPlay muted style={{ width: '640px', height: '480px' }} />
        <canvas ref={canvasRef} width="640" height="480" className="overlay" />
      </div>
      <button onClick={registerFace} style={{ padding: '10px', marginTop: '10px' }}>
        Register Face
      </button>
    </div>
  );
}

export default Attend;