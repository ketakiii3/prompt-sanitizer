import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sanitizePrompt = async (prompt, getLlmResponse = true) => {
  try {
    const response = await api.post('/sanitize', {
      prompt,
      get_llm_response: getLlmResponse,
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export default api;