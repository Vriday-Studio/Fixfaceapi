// src/ValuePrompt.jsx
import React, { useEffect, useState } from 'react';

const ValuePrompt = () => {
  const [promptValue, setPromptValue] = useState('');

  useEffect(() => {
    const promptLabel = document.getElementById("prompt");
    if (promptLabel) {
      setPromptValue(promptLabel.innerText);
    }
  }, []);

  return (
    <div>
      <h1>Nilai Prompt</h1>
      <p>{promptValue}</p>
    </div>
  );
};

export default ValuePrompt;


