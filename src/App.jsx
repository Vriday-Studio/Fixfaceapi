import {useRef,useEffect,useState} from 'react'
import './App.css'
import * as faceapi from 'face-api.js'
function App(){
  const videoRef = useRef()
  const canvasRef = useRef()
  const faceWidthInMeters = 0.15; // Average face width in meters
  const focalLength = 500; // Example focal length in pixels
  const [nama, setNama] = useState("unknown");
 
let iternoFaces=0;
  let totalDetectedFaces = 0;
  let numberOfFaces=0;
  let historySumarry="";  
  let historyGender="";  
  let historyAge=0;  
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
    const labels =["Raisa","Jokowi","JefriNichol","Vior"];
    return Promise.all(
      labels.map(async (label) => {
        const descriptions = [];
        for (let i = 1; i <= 4; i++) {
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
    var nametemp="Membaca Wajah...";
    let iterDetect=0;
   
  const faceMyDetect = ()=>{
    setInterval(async()=>{
      const detections = await faceapi.detectAllFaces(videoRef.current,
        new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions().withAgeAndGender().withFaceDescriptors();
       if(detections.length == 0){  
       
        iternoFaces++;
        if(iternoFaces>20){
          console.log("No face detected Longtime");
          iternoFaces=0;
          numberOfFaces=0;
        }
        console.log("No face detected");
       }else{
        iternoFaces=0;
        numberOfFaces=detections.length;
       }
        
       // console.log("Number of Faces: "+numberOfFaces);
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


    
   /*   
    const labeledFaceDescriptors = await getLabeledFaceDescription();
   // console.log("Labeled Face Descriptors:", labeledFaceDescriptors);

     const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);
    // console.log("Face Matcher created:", faceMatcher);
   
     if (detections.length > 0) {
      detections.forEach((detection) => {
        const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
        console.log(`Detected: ${bestMatch.toString()}`); 
        nametemp= "Wajah : "+ bestMatch.toString();
        if (!nametemp.includes("unknown")) {
          document.getElementById("prediksiwajah").innerHTML = nametemp;
        }
      //  setNama(bestMatch.toString());// Log the best match
      
      });
    }
      */
      // Draw age and gender
    
      detections.forEach((detection, index) => { iterDetect++;
        if(iterDetect>6){
          iterDetect=0;
        }
       
        const { age, gender } = detection;
        const box = detection.detection.box;
        const distance = (focalLength * faceWidthInMeters) / box.width;
        const indekDetect=index;
        //console.log("index: "+indekDetect);
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
     
        if (dataBody.rows.length > 10) {
           //Remove the oldest row
         dataBody.deleteRow(0);
     }
      // Calculate averages every 10 rows
      if (dataBody.rows.length % 10 === 0) {
        let totalAge = 0;
        let totalDistance = 0;
        let maleCount = 0;
        let femaleCount = 0;

        for (let i = 0; i < dataBody.rows.length; i++) {
          const row = dataBody.rows[i];
          const rowGender = row.cells[0].innerText;
          const rowAge = parseInt(row.cells[1].innerText);
          const rowDistance = parseFloat(row.cells[2].innerText);

          totalAge += rowAge;
          totalDistance += rowDistance;

          if (rowGender === "male") {
            maleCount++;
          } else if (rowGender === "female") {
            femaleCount++;
          }
        }

        const averageAge = totalAge / dataBody.rows.length;
        const averageDistance = totalDistance / dataBody.rows.length;
        const summarizedGender = maleCount > femaleCount ? "male" : "female";
        const promptLabel = document.getElementById("prompt");

        historyGender=summarizedGender;
        historyAge=averageAge;
     // Buat baris baru untuk dataBodyindex
     const newRowIndex = document.createElement("tr");
     newRowIndex.innerHTML = `
         <td style="border: 4px solid white; padding: 8px; text-align: center;">${indekDetect}</td>
         <td style="border: 4px solid white; padding: 8px; text-align: center;">${summarizedGender}</td>
         <td style="border: 4px solid white; padding: 8px; text-align: center;">${Math.round(averageAge)}</td>
         <td style="border: 4px solid white; padding: 8px; text-align: center;">${averageDistance.toFixed(2)}</td>
         <td style="border: 4px solid white; padding: 8px; text-align: center;">${promptLabel.innerText}</td>
     `;

     // Tambahkan baris baru ke dataBodyindex
     const dataBodyIndex = document.getElementById("dataBodyindex");
     if (newRowIndex !== undefined) {
  
         dataBodyIndex.appendChild(newRowIndex);
      
     }
     if (dataBodyIndex.rows.length > 5) {
      //Remove the oldest row
    dataBodyIndex.deleteRow(0);
}  
        //}
        // Update the summarize label
        const summarizeLabel = document.getElementById("summarize");
        const rtTotalWajahLabel = document.getElementById("rtTotalWajah");
        //console.log("iterDetect: "+iterDetect);
        rtTotalWajahLabel.innerText="Realtime number of faces: " +numberOfFaces;
        if(iterDetect==5){
        summarizeLabel.innerText = `Rata-rata: Gender: ${summarizedGender}, 
        Umur: ${Math.round(averageAge)}, Jarak: ${averageDistance.toFixed(2)}, 
        Total Wajah: ${numberOfFaces}`; // Tambahkan total wajah const promptLabel = document.getElementById("prompt");
        }
        historySumarry=summarizeLabel.innerText;
        promptLabel.innerText="";
        if (detections.length > 0) {
      //    setNumberOfFaces(detections.length);

          if (summarizedGender === "female") {
            if (averageAge < 12) {
              promptLabel.innerText = "Pengunjung masih wanita kecil muda berumur dibawah 12 tahun"; // Below 12 years
            } else if (averageAge < 30) {
              promptLabel.innerText = "Pengunjung wanita muda antara 13-30 tahun"; // Female under 30
            }else if (averageAge < 60) {
              promptLabel.innerText = "Pengunjung wanita dewasa antara 31-60 tahun"; // Male 30 to 60
        }  else {
              promptLabel.innerText = "Pengunjung wanita tua berumur diatas 60 tahun"; // Female 30 and above
            }
          } else { // Male
            if (averageAge < 12) {
              promptLabel.innerText = "Pengunjung masih pria kecil muda berumur dibawah 12 tahun"; // Below 12 years
            } else if (averageAge < 30) {
              promptLabel.innerText = "Pengunjung pria muda antara 13-30 tahun"; // Male under 30
            } else if (averageAge < 60) {
                  promptLabel.innerText = "Pengunjung pria dewasa antara 31-60 tahun"; // Male 30 to 60
            } else {
              promptLabel.innerText = "Pengunjung pria tua berumur diatas 60 tahun"; // Male 60 and above
            }
          }// Face detected
         
        } else {
          if(detections.length == 0){
          
   
             console.log("No face detected");
           }
            
           
          totalDetectedFaces=0;
          summarizeLabel.innerText = "..."
          promptLabel.innerText = "Mari kesini, aku AI yang bisa memandu di tempat ini"; // No face detected
        }
        promptLabel.innerText += " ,Orang didepan berjumlah:"+numberOfFaces;
      }
       
   //   const rows = Array.from(dataBody.rows);
     // console.log("data1"+row.cells[0].innerText);
      });
// Update the prompt label based on face detection

    },800);
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
           <label id="prediksiwajah"></label>
           <label id="promptFinal" style={{color: 'red'}}></label>
           <br></br>
           <br></br>
           Prompt Send:
           <br></br>
           <label id="prompt">...</label>
           <br></br>
           <br></br>
           <label id="rtTotalWajah">Realtime number of faces: 0</label>
           <br></br>
           <label >Summarize:</label>
           <label id="summarize">...</label>
           <label id="iskeluarga"></label>
                
           <table id="dataplayerindex" style={{ borderCollapse: 'collapse', width: '100%', textAlign: 'center', display: 'block' }}>
        <thead style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>
          <tr  style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>
          <th style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>Index</th>
            <th style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>Gender</th>
            <th style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>Umur</th>
            <th style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>Jarak (m)</th>
            <th style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>Prompt</th>
          </tr>
        </thead>
        <tbody id="dataBodyindex" style={{ border: '4px solid white', padding: '8px', textAlign: 'center' }}>
          {/* Rows will be added here dynamically */}
        </tbody>
        
      </table>     
      <table id="dataplayer" style={{ borderCollapse: 'collapse', width: '100%', textAlign: 'center', display: 'block' }}>
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
      
      </div>
      </div>
          </div>
          
  );
}

export default App;