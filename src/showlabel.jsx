// src/ShowLabel.js
import React, { useEffect } from 'react';

const ShowLabel = () => {
    const promptValue = localStorage.getItem("latestPrompt") || "No prompt stored yet";
    return promptValue;
     // This component does not render anything
};


export default ShowLabel;
