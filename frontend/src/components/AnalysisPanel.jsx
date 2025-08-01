// components/AnalysisPanel.jsx
import React from 'react';

const AnalysisPanel = ({ analysis }) => {
  const {
    original_prompt,
    rules_triggered,
    pii_detected,
    classifier_result,
    final_decision,
    sanitized_prompt,
    blocked
  } = analysis;

  const getDecisionColor = (decision) => {
    switch (decision) {
      case 'allowed': return '#28a745';
      case 'blocked': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'block': return '#dc3545';
      case 'warn': return '#ffc107';
      case 'sanitize': return '#17a2b8';
      default: return '#6c757d';
    }
  };

  return (
    <div className="analysis-panel">
      <h2>Analysis Results</h2>
      
      <div className="analysis-grid">
        {/* Original Prompt */}
        <div className="analysis-section">
          <h3>Original Prompt</h3>
          <div className="content-box">
            <p>{original_prompt}</p>
          </div>
        </div>

        {/* Rules Triggered / PII Detected */}
        <div className="analysis-section">
          <h3>Rules & PII Detection</h3>
          <div className="content-box">
            {rules_triggered.length > 0 && (
              <div className="rules-section">
                <h4>Rules Triggered:</h4>
                <ul>
                  {rules_triggered.map((rule, index) => (
                    <li key={index} className="rule-item">
                      <span 
                        className="rule-severity"
                        style={{ backgroundColor: getSeverityColor(rule.severity) }}
                      >
                        {rule.severity}
                      </span>
                      <strong>{rule.rule_name}:</strong> "{rule.matched_text}"
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {pii_detected.length > 0 && (
              <div className="pii-section">
                <h4>PII Detected & Redacted:</h4>
                <ul>
                  {pii_detected.map((pii, index) => (
                    <li key={index} className="pii-item">{pii}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {rules_triggered.length === 0 && pii_detected.length === 0 && (
              <p className="no-issues">No rules triggered or PII detected</p>
            )}
          </div>
        </div>

        {/* Classifier Result */}
        <div className="analysis-section">
          <h3>AI Classifier Result</h3>
          <div className="content-box">
            <div className="classifier-result">
              <div className="result-item">
                <strong>Label:</strong> 
                <span className={`label ${classifier_result.label}`}>
                  {classifier_result.label}
                </span>
              </div>
              <div className="result-item">
                <strong>Confidence:</strong> 
                <span className="confidence">
                  {(classifier_result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="result-item">
                <strong>Threshold:</strong> 
                <span className="threshold">
                  {(classifier_result.threshold * 100).toFixed(1)}%
                </span>
              </div>
              
              {/* Confidence Bar */}
              <div className="confidence-bar">
                <div className="confidence-bar-bg">
                  <div 
                    className="confidence-bar-fill"
                    style={{ 
                      width: `${classifier_result.confidence * 100}%`,
                      backgroundColor: classifier_result.label === 'harmful' ? '#dc3545' : '#28a745'
                    }}
                  ></div>
                  <div 
                    className="threshold-line"
                    style={{ left: `${classifier_result.threshold * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Final Decision */}
        <div className="analysis-section">
          <h3>Final Decision</h3>
          <div className="content-box">
            <div className="final-decision">
              <div 
                className={`decision-badge ${final_decision}`}
                style={{ backgroundColor: getDecisionColor(final_decision) }}
              >
                {final_decision.toUpperCase()}
              </div>
              
              <div className="sanitized-prompt">
                <h4>Sanitized Prompt:</h4>
                <div className={`prompt-result ${blocked ? 'blocked' : 'allowed'}`}>
                  {sanitized_prompt}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisPanel;