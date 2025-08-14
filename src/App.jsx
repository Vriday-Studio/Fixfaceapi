import { Route, Routes, useNavigate } from "react-router-dom";
import ShowLabel from './showlabel';
import Camera from './Camera';// your face detection code
import MainApp from './MainApp'; // your face detection code
function App() {
  const navigate = useNavigate();

  return (
    <div className="myapp">
      <Routes>
        <Route path="/" element={<MainApp/>} />
        <Route path="/showlabel" element={<ShowLabel />} />
        <Route path="/camera" element={<Camera />} />
      </Routes>
    </div>
  );
}

export default App;
