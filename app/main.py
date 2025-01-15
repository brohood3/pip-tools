from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with minimal configuration
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
openai_client = OpenAI(
    api_key=api_key,
    http_client=None,  # Explicitly disable custom http client
    timeout=60.0  # Set OpenAI client timeout to 60 seconds
)

app = FastAPI(
    title="Trading Tools API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PromptRequest(BaseModel):
    prompt: str

class ToolSelector:
    def run(self, prompt: str) -> Dict[str, Any]:
        from tools.tool_selector.tool import get_tool_for_prompt
        return get_tool_for_prompt(openai_client, prompt)

class QueryExtract:
    def __init__(self):
        """Initialize with OpenAI client"""
        from tools.query_extract.tool import QueryExtract as QueryExtractTool
        self.tool = QueryExtractTool()

    def run(self, prompt: str) -> Dict[str, Any]:
        return self.tool.run(prompt)

class FundamentalAnalysis:
    def __init__(self):
        """Initialize with OpenAI client"""
        from tools.fundamental_analysis.tool import FundamentalAnalysis as FATool
        self.tool = FATool()

    def run(self, prompt: str) -> Dict[str, Any]:
        return self.tool.run(prompt)

class TechnicalAnalysis:
    def __init__(self):
        """Initialize with OpenAI client"""
        from tools.ta_analysis.tool import TechnicalAnalysis as TATool
        self.tool = TATool()

    def run(self, prompt: str) -> Dict[str, Any]:
        return self.tool.run(prompt)

class FuturePredictor:
    def __init__(self):
        """Initialize with OpenAI client"""
        from tools.future_predictor.tool import FuturePredictor as FPTool
        self.tool = FPTool()

    def run(self, prompt: str) -> Dict[str, Any]:
        return self.tool.run(prompt)

class TechnicalAnalysisEssential:
    def __init__(self):
        """Initialize with OpenAI client"""
        from tools.ta_essential.tool import TechnicalAnalysis as TAEssentialTool
        self.tool = TAEssentialTool()

    def run(self, prompt: str) -> Dict[str, Any]:
        return self.tool.run(prompt)

@app.get("/")
async def root():
    return {"message": "Trading Tools API is running"}

@app.get("/tools")
async def list_tools():
    """List all available tools with their descriptions"""
    return {
        "tools": [
            {
                "name": "tool-selector",
                "description": "Selects the most appropriate tool for a given analysis task",
                "endpoint": "/tool-selector"
            },
            {
                "name": "query-extract",
                "description": "Extracts structured information from natural language queries",
                "endpoint": "/query-extract"
            },
            {
                "name": "fundamental-analysis",
                "description": "Performs fundamental analysis on cryptocurrency tokens",
                "endpoint": "/fundamental-analysis"
            },
            {
                "name": "technical-analysis",
                "description": "Performs technical analysis on trading pairs",
                "endpoint": "/ta-analysis"
            },
            {
                "name": "future-predictor",
                "description": "Generates detailed predictions for cryptocurrency and market-related questions",
                "endpoint": "/future-predictor"
            },
            {
                "name": "ta-essential",
                "description": "Essential technical analysis using key indicators for cryptocurrency market analysis",
                "endpoint": "/ta-essential"
            }
        ]
    }

@app.post("/tool-selector")
async def tool_selector(request: PromptRequest):
    try:
        tool = ToolSelector()
        result = tool.run(request.prompt)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-extract")
async def query_extract(request: PromptRequest):
    try:
        tool = QueryExtract()
        result = tool.run(request.prompt)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fundamental-analysis")
async def fundamental_analysis(request: PromptRequest):
    try:
        tool = FundamentalAnalysis()
        result = tool.run(request.prompt)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ta-analysis")
async def technical_analysis(request: PromptRequest):
    try:
        tool = TechnicalAnalysis()
        result = tool.run(request.prompt)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/future-predictor")
async def future_predictor(request: PromptRequest, response: Response):
    try:
        # Set a longer timeout for this specific endpoint
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Connection"] = "keep-alive"
        
        tool = FuturePredictor()
        result = tool.run(request.prompt)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ta-essential")
async def ta_essential(request: PromptRequest):
    try:
        tool = TechnicalAnalysisEssential()
        result = tool.run(request.prompt)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 