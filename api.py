"""
FastAPI server for fundamental analysis endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from fundamental_analysis import (
    get_coingecko_id_from_prompt,
    get_token_details,
    get_investment_analysis,
    get_project_research,
    get_market_context_analysis,
    generate_investment_report
)

app = FastAPI(title="Crypto Fundamental Analysis API")

class AnalysisRequest(BaseModel):
    prompt: str

class AnalysisResponse(BaseModel):
    token_id: str
    token_name: str
    token_symbol: str
    report: str
    tokenomics_analysis: Optional[str]
    project_research: Optional[str]
    market_context: Optional[str]

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_token(request: AnalysisRequest):
    """Generate fundamental analysis for a token based on a prompt."""
    try:
        # Get CoinGecko ID from prompt
        token_id = get_coingecko_id_from_prompt(request.prompt)
        if not token_id:
            raise HTTPException(status_code=400, detail="Could not determine which token to analyze")
        
        # Get token details
        token_details = get_token_details(token_id)
        if not token_details:
            raise HTTPException(status_code=404, detail=f"Could not fetch details for token {token_id}")
        
        # Generate analyses
        tokenomics_analysis = get_investment_analysis(token_details)
        project_research = get_project_research(token_details)
        market_context = get_market_context_analysis(token_details)
        
        if not all([tokenomics_analysis, project_research, market_context]):
            raise HTTPException(status_code=500, detail="Error generating complete analysis")
        
        # Generate final report
        final_report = generate_investment_report(
            token_details,
            tokenomics_analysis,
            project_research,
            market_context
        )
        
        if not final_report:
            raise HTTPException(status_code=500, detail="Error generating final report")
        
        return AnalysisResponse(
            token_id=token_id,
            token_name=token_details["name"],
            token_symbol=token_details["symbol"],
            report=final_report,
            tokenomics_analysis=tokenomics_analysis,
            project_research=project_research,
            market_context=market_context
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 