# services/llm_service.py
import openai
from ..config import Config
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
    
    async def get_response(self, prompt: str) -> str:
        """
        Get response from OpenAI GPT model
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide clear, informative, and safe responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return "Sorry, I couldn't generate a response at this time. Please try again later."
