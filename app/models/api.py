from pydantic import BaseModel
from typing import Optional


class PromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
