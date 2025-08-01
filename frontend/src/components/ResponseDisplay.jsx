// components/ResponseDisplay.jsx
import React from 'react';

const ResponseDisplay = ({ response }) => {
  if (!response) return null;

  return (
    <div className="response-display">
      <h2>LLM Response</h2>
      <div className="response-content">
        <div className="response-box">
          <p>{response}</p>
        </div>
      </div>
    </div>
  );
};

export default ResponseDisplay;