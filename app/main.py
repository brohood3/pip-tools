from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.models.api import PromptRequest
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


@app.post("/{tool_name}")
async def run_tool(tool_name: str, request: PromptRequest):
    if tool_name not in TOOL_TO_MODULE:
        raise ToolNotFoundError(tool_name)

    tool = TOOL_TO_MODULE[tool_name]
    result = tool.run(request.prompt)

    return {"result": result}
