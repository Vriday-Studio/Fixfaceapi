// src/DisplayPrompt.jsx
import React from 'react';
import { useLocation } from 'react-router-dom';

const DisplayPrompt = () => {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const promptValue = queryParams.get('promptValue') || 'No value provided';

  return (
    <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Prompt Value</h1>
      <p>{decodeURIComponent(promptValue)}</p>
    </div>
  );
};

export default DisplayPrompt;