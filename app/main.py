from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from io import BytesIO
import base64

from app.models.api import PromptRequest, ToolSelectorRequest
from app.tools.helpers import TOOLS, TOOL_TO_MODULE
from app.utils.errors import ToolNotFoundError, AppError
from app.utils.logging import logger

app = FastAPI(title="Trading Tools API", docs_url="/docs", redoc_url="/redoc")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception occurred: {exc}")

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},  # Generic message for security
    )

@app.exception_handler(AppError)
async def generic_exception_handler(request: Request, exc: AppError):
    logger.exception(f"Unhandled exception occurred: {exc}")

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )



@app.get("/")
async def root():
    return {"message": "Trading Tools API is running"}


@app.get("/tools")
async def list_tools():
    """List all available tools with their descriptions"""
    return dict(tools=TOOLS)


@app.post("/tool_selector")
async def run_tool_selector(request: ToolSelectorRequest):
    """Special endpoint for tool selector that supports filtering available tools"""
    tool = TOOL_TO_MODULE["tool_selector"]
    result = tool.run(request.prompt, request.system_prompt, allowed_tools=request.allowed_tools)
    return {"result": result}

@app.post("/{tool_name}")
async def run_tool(tool_name: str, prompt_request: PromptRequest, request: Request):
    """Generic endpoint for all other tools"""
    # Case-insensitive lookup
    tool_name_lower = tool_name.lower()
    available_tools = {k.lower(): v for k, v in TOOL_TO_MODULE.items()}
    
    if tool_name_lower not in available_tools:
        raise ToolNotFoundError(tool_name)

    tool = available_tools[tool_name_lower]
    result = tool.run(prompt_request.prompt, prompt_request.system_prompt)
    
    # Handle chart data for tools that generate charts
    if tool_name_lower in ["technical_analysis", "synth_chart_generator"] and "chart" in result:
        chart_data = result.pop("chart")  # Remove chart from main response
        if chart_data:  # Only add chart_url if we actually have a chart
            base_url = str(request.base_url).rstrip('/')
            result["chart_url"] = f"{base_url}/chart/latest"  # Full URL
    
    return result  # Return the raw result

@app.get("/chart/latest")
async def get_latest_chart():
    """Serve the latest generated chart image."""
    try:
        # Try to get chart from technical analysis tool first
        tool = TOOL_TO_MODULE.get("technical_analysis")
        if tool:
            chart_data = tool.get_latest_chart()
            if chart_data:
                # Process and return the chart
                if "base64," in chart_data:
                    chart_data = chart_data.split("base64,")[1]
                
                try:
                    image_bytes = base64.b64decode(chart_data)
                    headers = {
                        "Cache-Control": "public, max-age=300",  # Cache for 5 minutes
                        "ETag": str(hash(chart_data))  # Add ETag for caching
                    }
                    return StreamingResponse(
                        BytesIO(image_bytes), 
                        media_type="image/png",
                        headers=headers
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail="Invalid chart data")
        
        # If no technical analysis chart, try synth chart generator
        tool = TOOL_TO_MODULE.get("synth_chart_generator")
        if tool:
            chart_data = tool.get_latest_chart()
            if chart_data:
                # Process and return the chart
                if "base64," in chart_data:
                    chart_data = chart_data.split("base64,")[1]
                
                try:
                    image_bytes = base64.b64decode(chart_data)
                    headers = {
                        "Cache-Control": "public, max-age=300",  # Cache for 5 minutes
                        "ETag": str(hash(chart_data))  # Add ETag for caching
                    }
                    return StreamingResponse(
                        BytesIO(image_bytes), 
                        media_type="image/png",
                        headers=headers
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail="Invalid chart data")
        
        raise HTTPException(status_code=404, detail="No chart available")
            
    except Exception as e:
        logger.exception("Error serving chart")
        raise HTTPException(status_code=500, detail=str(e))
