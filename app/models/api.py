from pydantic import BaseModel
from typing import Optional, List


class PromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None  # Optional model parameter to configure OpenAI model

class ToolSelectorRequest(PromptRequest):
    allowed_tools: Optional[List[str]] = None  # If None, all tools are available
