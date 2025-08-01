import React, { useState } from 'react';
import PromptInput from './components/PromptInput';
import AnalysisPanel from './components/AnalysisPanel';
import ResponseDisplay from './components/ResponseDisplay';
import { sanitizePrompt } from './services/api';
import './App.css';

function App() {
  const [analysis, setAnalysis] = useState(null);
  const [llmResponse, setLlmResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (prompt, getLlmResponse) => {
    setLoading(true);
    setError(null);
    setAnalysis(null);
    setLlmResponse(null);

    try {
      const result = await sanitizePrompt(prompt, getLlmResponse);
      setAnalysis(result.analysis);
      setLlmResponse(result.llm_response);
    } catch (err) {
      setError('Failed to process prompt. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Safety-Aware Prompt Sanitizer</h1>
        <p>Enter your prompt below to analyze and sanitize it before sending to an LLM</p>
      </header>

      <main className="App-main">
        <div className="container">
          <PromptInput onSubmit={handleSubmit} loading={loading} />
          
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {analysis && (
            <AnalysisPanel analysis={analysis} />
          )}

          {llmResponse && (
            <ResponseDisplay response={llmResponse} />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;