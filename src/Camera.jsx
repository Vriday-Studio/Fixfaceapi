import {useRef,useEffect,useState} from 'react'
import './App.css'
import * as faceapi from 'face-api.js'
import OpenAI from "openai";
import { getResponse } from './example.mjs';
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
function Camera(){
  const videoRef = useRef()
  const canvasRef = useRef()
  const faceWidthInMeters = 0.15; // Average face width in meters
  const focalLength = 500; // Example focal length in pixels
  const [nama, setNama] = useState("unknown");
  
  // State untuk API Key dan visibilitas pop-up
  const [apiKey, setApiKey] = useState('');
  const [showPopup, setShowPopup] = useState(true);
const defaultProminit="Anda berperan sebagai Pemandu Pengunjung di Galeri Indonesia Kaya, Nama kamu adalah IKA, Sambutlah pengunjung sesuai informasi yang nanti akan saya kirimkan dengan ramah,  ,  jika mengerti  jawab dengan sapaan: Hai, namaku IKA, saya bisa membantu anda disini!";
  
  const defPromptNoPeople="Tak ada orang disini, buat kalimat menarik dan trivia tentang kebudayaaan indonesia yang bertujuan orang yang agak jauh agar mendekat";
  const defPromptIdentify="Sapalah pengunjung berikut ini, sertakan panggilan orang berdasarkan gender dan usia, dan Tambahkan basa basi seperti: trivia pengetahuan unik budaya indonesia atau topik cuaca atau kemacetan  atau keadaan di Galeri Indonesia Kaya atau lain sebagainya, dan bedakan berdasar jumlah orang, misal sendiri kamu bisa tambahkan kalimat: sendirian aja nih?, kalau berdua: wah, kalian temenan atau saudara nih?  kalo banyak: wah rombongan nih, silahkan kamu boleh improve kalimat basa basi yang lain, usahakan agar tiap saya kirim informasi kalimatnya berbeda,  berikut info yang tersedia: ";
  const cekppd = ()=>{
    if(localStorage.getItem('panggilanPerdetik') == null){
      return 11000;
    }else{
      return localStorage.getItem('panggilanPerdetik');
    }
  }
  const cekinit = ()=>{
    if(localStorage.getItem('promptinit') == null){
      return defaultProminit;
    }else{
      return localStorage.getItem('promptinit');
    }
  }
  const cekNopeople = ()=>{
    if(localStorage.getItem('promptNoPeople') == null){
      return defPromptNoPeople;
    }else{
      return localStorage.getItem('promptNoPeople');
    }
  }
  const cekIdentify = ()=>{
    if(localStorage.getItem('promptIdentify') == null){
      return defPromptIdentify;
    }else{
      return localStorage.getItem('promptIdentify');
    }
  }
  
  // State untuk variabel input
  const [panggilanPerdetik, setPanggilanPerdetik] = useState(cekppd);
  const [promptinit, setPromptInit] = useState(cekinit);
  const [promptNoPeople, setPromptNoPeople] = useState(cekNopeople);
  const [promptIdentify,setPromptIdentify] = useState(cekIdentify);
  // Mengambil nilai dari localStorage saat komponen dimuat
  useEffect(() => {
    const storedPanggilanPerdetik = localStorage.getItem('panggilanPerdetik');
    const storedPromptInit = localStorage.getItem('promptinit');
    const storedPromptNoPeople = localStorage.getItem('promptNoPeople');
    const storedPromptIdentify = localStorage.getItem('promptIdentify');

    if (storedPanggilanPerdetik) {
      setPanggilanPerdetik(Number(storedPanggilanPerdetik));
    }
    if (storedPromptInit) {
      setPromptInit(storedPromptInit);
    }
    if (storedPromptNoPeople) {
      setPromptNoPeople(storedPromptNoPeople);
    }
    if (storedPromptIdentify) {
      setPromptIdentify(storedPromptIdentify);
    }
    console.log("get item storedPanggilanPerdetik ="+storedPanggilanPerdetik );
  }, []);

  // Menyimpan nilai ke localStorage setiap kali ada perubahan
  useEffect(() => {
    console.log("Set item storedPanggilanPerdetik ="+ panggilanPerdetik );
    localStorage.setItem('panggilanPerdetik', panggilanPerdetik);
    localStorage.setItem('promptinit', promptinit);
    localStorage.setItem('promptNoPeople', promptNoPeople);
    localStorage.setItem('promptIdentify', promptIdentify);
  }, [panggilanPerdetik, promptinit, promptNoPeople, promptIdentify]);

  // Mengambil API Key dari localStorage saat komponen dimuat
  useEffect(() => {
    const storedApiKey = localStorage.getItem('apiKey');
    if (storedApiKey) {
      setApiKey(storedApiKey);
    }
  }, []);

  // Fungsi untuk menyimpan API Key dan menutup pop-up
  const handleSaveApiKey = () => {
    localStorage.setItem('apiKey', apiKey);
    setShowPopup(false);
  };
 
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
//let promptinit=  "Anda berperan sebagai Pemandu Pengunjung di Galeri Indonesia Kaya, Nama kamu adalah IKA, Sambutlah pengunjung sesuai informasi yang nanti akan saya kirimkan dengan ramah,  ,  jika mengerti  jawab dengan sapaan: Hai, namaku IKA, saya bisa membantu anda disini!";

let panggilAI=false;
  let promptKirim = promptinit;
  let jumlahpanggilan=0;
// Fungsi untuk mendapatkan respons
  const fetchResponse = async () => {
    if(panggilAI){
   // Ganti dengan prompt yang Anda inginkan
jumlahpanggilan++;
console.log("Jumlah Call:"+jumlahpanggilan );
    const promptFinale = document.getElementById("promptFinal");
    const response = await getResponse(promptKirim);
 
    promptFinale.innerText=response;
  }else{
    console.log("AI tak dipanggil");
    const promptFinale = document.getElementById("promptFinal");
    promptFinale.innerText="Hai Namaku IKA, aku pemandu disini"
  } // Tampilkan respons di konsol
};
//let panggilanPerdetik=10000;
// Panggil fetchResponse setiap 10 detik
useEffect(() => {
   fetchResponse(); // Panggil sekali saat komponen dimuat
    const intervalId = setInterval(fetchResponse, panggilanPerdetik); // 10000 ms = 10 detik

    return () => clearInterval(intervalId); // Bersihkan interval saat komponen di-unmount
}, []);


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
      //disable panggil AI OpenAI here
    //panggilAI=true;
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
  // let promptNoPeople="Tak ada orang disini, buat kalimat menarik dan trivia tentang kebudayaaan indonesia yang bertujuan orang yang agak jauh agar mendekat";
 //let promptIdentify="Sapalah pengunjung berikut ini, sertakan panggilan orang berdasarkan gender dan usia, dan Tambahkan basa basi seperti: trivia pengetahuan unik budaya indonesia atau topik cuaca atau kemacetan  atau keadaan di Galeri Indonesia Kaya atau lain sebagainya, dan bedakan berdasar jumlah orang, misal sendiri kamu bisa tambahkan kalimat: sendirian aja nih?, kalau berdua: wah, kalian temenan atau saudara nih?  kalo banyak: wah rombongan nih, silahkan kamu boleh improve kalimat basa basi yang lain, usahakan agar tiap saya kirim informasi kalimatnya berbeda,  berikut info yang tersedia: ";
   const faceMyDetect = ()=>{
    setInterval(async()=>{
      const detections = await faceapi.detectAllFaces(videoRef.current,
        new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions().withAgeAndGender().withFaceDescriptors();
       if(detections.length == 0){  
       
        iternoFaces++;
        if(iternoFaces>20){
          const promptLabelx = document.getElementById("prompt");
          //console.log("No face detected Longtime");
          promptKirim=promptNoPeople;
          promptLabelx.innerText = "Tak ada orang disini, kirim trivia";
        //  panggilAI=false;
          iternoFaces=0;
          numberOfFaces=0;
        }
      //  console.log("No face detected");
       }else{
        iternoFaces=0;
        numberOfFaces=detections.length;
       // panggilAI=true;
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
        let indexTempDet= 99;
        if (detections.length >=1) {
          promptLabel.innerText="";
      //    setNumberOfFaces(detections.length);
      indexTempDet=indekDetect;
          console.log("indekdetek ="+indekDetect);
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
         
        }
        
        else {
          if(detections.length == 0){
          
   
             console.log("No face detected");
           }
            
           
          totalDetectedFaces=0;
          summarizeLabel.innerText = "..."
          promptLabel.innerText = promptNoPeople; // No face detected
        }
        promptLabel.innerText += " ,Orang didepan berjumlah:"+numberOfFaces;
        localStorage.setItem("latestPrompt", promptLabel.innerText);

        
        promptKirim= promptIdentify+promptLabel.innerText;
        console.log("prompt kirim="+promptKirim);
      }
       
   //   const rows = Array.from(dataBody.rows);
     // console.log("data1"+row.cells[0].innerText);
      });
// Update the prompt label based on face detection

    },5800);
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
 
      {/* Pop-up untuk memasukkan API Key */}
      {showPopup && (
        <div className="popup">
          <div className="popup-content">
            <h4 style={{ color: 'blue' }}>Masukkan API Key</h4>
            <textarea 
              rows="4" // Menentukan jumlah baris
              cols="50" // Menentukan jumlah kolom
              value={apiKey} 
              onChange={(e) => setApiKey(e.target.value)} 
              placeholder="Masukkan API Key Anda" 
            />
            <button onClick={handleSaveApiKey}>OK</button>
          </div>
        </div>
      )}

      <div className="appvide">
      
      <video crossOrigin="anonymous" ref={videoRef} autoPlay></video>
      </div>
      <canvas ref={canvasRef} width="940" height="650"
      className="appcanvas"/>
           <div>
           <label id="prediksiwajah"></label>
           Prompt Chat AI Read:
           <br></br>
           <label id="promptFinal" style={{color: 'yellow'}}></label>
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
      <table id="dataplayer" style={{ borderCollapse: 'collapse', width: '100%', textAlign: 'center', display: 'none' }}>
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
       {/* Box isian untuk variabel */}
       <div>
        <label>Panggilan Per Detik:</label>
        <input 
          type="number" 
          value={panggilanPerdetik} 
          onChange={(e) => setPanggilanPerdetik(e.target.value)} 
        />
      </div>
      <div>
        <label>Prompt Init:</label>
        <textarea 
          rows="15" 
          cols="50" 
          value={promptinit} 
          onChange={(e) => setPromptInit(e.target.value)} 
        />
      </div>
      <div>
        <label>Prompt No People:</label>
        <textarea 
          rows="15" 
          cols="50" 
          value={promptNoPeople} 
          onChange={(e) => setPromptNoPeople(e.target.value)} 
        />
      </div>
      <div>
        <label>Prompt Identify:</label>
        <textarea 
          rows="15" 
          cols="50" 
          value={promptIdentify} 
          onChange={(e) => setPromptIdentify(e.target.value)} 
        />
      </div>

      </div>
          </div>
          
  );
}

export default Camera;