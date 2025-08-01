# Enhanced main.py with logging and metrics
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .services.sanitizer_service import SanitizerService
from .services.llm_service import LLMService
import logging
import time
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sanitizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Safety-Aware Prompt Sanitizer", 
    version="1.0.0",
    description="AI-powered prompt sanitizer with rule-based filtering and ML classification"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
sanitizer_service = SanitizerService()
llm_service = LLMService()

# Metrics storage (in production, use proper database)
metrics = {
    "total_requests": 0,
    "blocked_requests": 0,
    "allowed_requests": 0,
    "rule_blocks": 0,
    "ai_blocks": 0,
    "response_times": []
}

class PromptRequest(BaseModel):
    prompt: str
    get_llm_response: bool = True

class SanitizeResponse(BaseModel):
    analysis: dict
    llm_response: str = None
    processing_time: float = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

@app.post("/sanitize", response_model=SanitizeResponse)
async def sanitize_prompt(request: PromptRequest):
    start_time = time.time()
    
    try:
        # Update metrics
        metrics["total_requests"] += 1
        
        # Sanitize the prompt
        analysis = sanitizer_service.sanitize_prompt(request.prompt)
        
        # Update block metrics
        if analysis["blocked"]:
            metrics["blocked_requests"] += 1
            if analysis["rules_triggered"]:
                # Check if blocked by rules vs AI
                rule_block = any(rule["severity"] == "block" for rule in analysis["rules_triggered"])
                if rule_block:
                    metrics["rule_blocks"] += 1
                else:
                    metrics["ai_blocks"] += 1
            else:
                metrics["ai_blocks"] += 1
        else:
            metrics["allowed_requests"] += 1
        
        llm_response = None
        if request.get_llm_response and not analysis["blocked"]:
            # Get LLM response for safe prompts
            llm_response = await llm_service.get_response(analysis["sanitized_prompt"])
        
        processing_time = time.time() - start_time
        metrics["response_times"].append(processing_time)
        
        # Log the analysis
        logger.info(f"Prompt processed - Decision: {analysis['final_decision']} - Time: {processing_time:.3f}s")
        
        return SanitizeResponse(
            analysis=analysis,
            llm_response=llm_response,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    avg_response_time = sum(metrics["response_times"]) / len(metrics["response_times"]) if metrics["response_times"] else 0
    
    return {
        "total_requests": metrics["total_requests"],
        "blocked_requests": metrics["blocked_requests"],
        "allowed_requests": metrics["allowed_requests"],
        "block_rate": metrics["blocked_requests"] / max(metrics["total_requests"], 1),
        "rule_blocks": metrics["rule_blocks"],
        "ai_blocks": metrics["ai_blocks"],
        "average_response_time": avg_response_time,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    return {
        "message": "Safety-Aware Prompt Sanitizer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    import uvicorn
    from .config import Config
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)