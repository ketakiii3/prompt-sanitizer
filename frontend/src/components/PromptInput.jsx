import React, { useState } from 'react';

const PromptInput = ({ onSubmit, loading }) => {
  const [prompt, setPrompt] = useState('');
  const [getLlmResponse, setGetLlmResponse] = useState(true);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (prompt.trim()) {
      onSubmit(prompt, getLlmResponse);
    }
  };

  return (
    <div className="prompt-input-section">
      <h2>Input Prompt</h2>
      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt here..."
            rows={6}
            className="prompt-textarea"
            disabled={loading}
          />
        </div>
        
        <div className="options">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={getLlmResponse}
              onChange={(e) => setGetLlmResponse(e.target.checked)}
              disabled={loading}
            />
            Get LLM Response (if prompt is safe)
          </label>
        </div>
        
        <button 
          type="submit" 
          className="submit-btn"
          disabled={loading || !prompt.trim()}
        >
          {loading ? 'Processing...' : 'Sanitize & Submit'}
        </button>
      </form>
    </div>
  );
};

export default PromptInput;