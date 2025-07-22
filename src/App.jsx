import {useRef,useEffect,useState} from 'react'
import './App.css'
import * as faceapi from 'face-api.js'

function App(){
  const videoRef = useRef()
  const canvasRef = useRef()
  const faceWidthInMeters = 0.15; // Average face width in meters
  const focalLength = 500; // Example focal length in pixels
  const [nama, setNama] = useState("unknown");
  // LOAD FROM USEEFFECT
  useEffect(() => {
    console.log("App component mounted");
  }, []);
  useEffect(()=>{
    startVideo()
    videoRef && loadModels()

  },[])



  // OPEN YOU FACE WEBCAM
  const startVideo = ()=>{
    navigator.mediaDevices.getUserMedia({video:true})
    .then((currentStream)=>{
      videoRef.current.srcObject = currentStream
    })
    .catch((err)=>{
      console.log(err)
    })
  }
  // LOAD MODELS FROM FACE API
  const loadModels = ()=>{
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
     faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
      faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
      faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
      faceapi.nets.faceExpressionNet.loadFromUri("/models"),
      faceapi.nets.ageGenderNet.loadFromUri("/models")
    ]).then(()=>{
      faceMyDetect();
    })
  }
  const getLabeledFaceDescription = () => {
    const labels =["Raisa","Jokowi","JefriNichol"];
    return Promise.all(
      labels.map(async (label) => {
        const descriptions = [];
        for (let i = 1; i <= 2; i++) {
          const img = await faceapi.fetchImage(`./labels/${label}/${i}.jpg`);
         // console.log("Image loaded:"+ `${label}/${i}.jpg`);
          const detections = await faceapi
            .detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();
       // console.log("Detections image :"+i);
          descriptions.push(detections.descriptor);
        }
      //  setNama(label);
        return new faceapi.LabeledFaceDescriptors(label, descriptions);
      })
    );
    }

  const faceMyDetect = ()=>{
    setInterval(async()=>{
      const detections = await faceapi.detectAllFaces(videoRef.current,
        new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions().withAgeAndGender().withFaceDescriptors();

      // DRAW YOU FACE IN WEBCAM
      canvasRef.current.innerHtml = faceapi.createCanvasFromMedia(videoRef.current);
      faceapi.matchDimensions(canvasRef.current,{ 
        width:940,
        height:650
      });
   //   const canvas = document.getElementById('canvasref');
   //   canvas.style.left
      const resized = faceapi.resizeResults(detections,{ 
         width:940,
        height:650
      });

      faceapi.draw.drawDetections(canvasRef.current,resized);
      faceapi.draw.drawFaceLandmarks(canvasRef.current,resized);
      faceapi.draw.drawFaceExpressions(canvasRef.current,resized);

      //const resizedDetections = faceapi.resizeResults(detections, displaySize);
      
    const labeledFaceDescriptors = await getLabeledFaceDescription();
   // console.log("Labeled Face Descriptors:", labeledFaceDescriptors);

     const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);
    // console.log("Face Matcher created:", faceMatcher);
    var nametemp="none";
     if (detections.length > 0) {
      detections.forEach((detection) => {
        const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
        console.log(`Detected: ${bestMatch.toString()}`); 
        setNama(bestMatch.toString());// Log the best match
        nametemp=bestMatch.toString();
      });
    }
      // Draw age and gender
      detections.forEach((detection, index) => {
        const { age, gender } = detection;
        const box = detection.detection.box;
        const distance = (focalLength * faceWidthInMeters) / box.width;
        
        // Create a new row
        const newRow = document.createElement("tr");
        newRow.innerHTML = `
            <td style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>${gender}</td>
            <td style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>${Math.round(age)}</td>
            <td style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>${distance.toFixed(2)}</td>
            <td style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>${nametemp}</td>
        `;

        // Append the new row to the table body
        const dataBody = document.getElementById("dataBody");
        if(newRow !== undefined){
          dataBody.appendChild(newRow);
        }
     

        // Check if there are more than 5 rows
    

        // Check every 5 detected rows
      /*  if ((index + 1) % 5 === 0) {
            const rows = Array.from(dataBody.rows);
            const selectedRows = rows.slice(-5); // Get the last 5 rows

            // Check if the selected rows have the same gender and age difference
            const sameGender = selectedRows.every(row => row.cells[0].innerText === gender);
            const ageDifferences = selectedRows.map(row => Math.abs(parseInt(row.cells[1].innerText) - Math.round(age)));

            if (sameGender && ageDifferences.every(diff => diff <= 5)) {
                // Remove the two oldest rows
                for (let i = 0; i < 2; i++) {
                    if (dataBody.rows.length > 0) {
                        dataBody.deleteRow(dataBody.rows.length - 1);
                    }
                }
            }

           
        }*/
        if (dataBody.rows.length > 5) {
           //Remove the oldest row
         dataBody.deleteRow(0);
     }
   //   const rows = Array.from(dataBody.rows);
     // console.log("data1"+row.cells[0].innerText);
      });

    },1000);
  }


  const registerFace = () => {
    console.log("Register Face clicked"); // Debugging line
    console.log("Is face detected:", isFaceDetected); // Check if a face is detected
    console.log("Models loaded:", modelsLoaded); // Check if models are loaded

    if (isFaceDetected) {
      console.log("Face registered!"); // Logic to register the face
      // Here you would typically call your face recognition API or logic
    } else {
      alert("No face detected. Please ensure your face is visible.");
    }
  };


  return (
    <div className="myapp">
      <h3>Deteksi Wajah</h3>
 
      <div className="appvide">
      
      <video crossOrigin="anonymous" ref={videoRef} autoPlay></video>
      </div>
      <canvas ref={canvasRef} width="940" height="650"
      className="appcanvas"/>
           <div>
           <label id="iskeluarga">Tabel</label>
      <table id="dataplayer" style={{ borderCollapse: 'collapse', width: '100%', textAlign: 'center' }}>
        <thead style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>
          <tr  style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>
            <th style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>Gender</th>
            <th style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>Umur</th>
            <th style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>Jarak (m)</th>
            <th style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>Nama</th>
          </tr>
        </thead>
        <tbody id="dataBody" style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>
          {/* Rows will be added here dynamically */}
        </tbody>
      </table>
      <div style={{ marginTop: '20px', textAlign: 'center' }}>
        <input
          type="text"
          id="faceName"
          placeholder="Enter name for face recognition"
          style={{ padding: '10px', width: '200px' }}
        />
        <button onClick={registerFace} style={{ padding: '10px', marginLeft: '10px' }}>
          Register Face
        </button>
      </div>
      </div>
          </div>
          
  );
}

export default App;