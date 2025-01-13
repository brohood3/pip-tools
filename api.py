"""
FastAPI server for fundamental analysis endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
import uvicorn
import fundamental_analysis_optimistic as optimistic
import fundamental_analysis_balanced as balanced

app = FastAPI(title="Crypto Fundamental Analysis API")

class AnalysisRequest(BaseModel):
    prompt: str
    style: Literal["optimistic", "balanced"] = "balanced"
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Analyze Bitcoin fundamentals",
                "style": "balanced"
            }
        }

class AnalysisResponse(BaseModel):
    token_id: str
    token_name: str
    token_symbol: str
    report: str
    tokenomics_analysis: Optional[str]
    project_research: Optional[str]
    market_context: Optional[str]
    analysis_style: str

@app.post("/fundamental-analysis", response_model=AnalysisResponse)
async def analyze_token(request: AnalysisRequest):
    """Generate fundamental analysis for a token based on a prompt.
    
    - style: Choose between 'optimistic' or 'balanced' analysis approach
    """
    try:
        # Select analysis module based on style
        analysis = optimistic if request.style == "optimistic" else balanced
        
        # Get CoinGecko ID from prompt
        token_id = analysis.get_coingecko_id_from_prompt(request.prompt)
        if not token_id:
            raise HTTPException(status_code=400, detail="Could not determine which token to analyze")
        
        # Get token details
        token_details = analysis.get_token_details(token_id)
        if not token_details:
            raise HTTPException(status_code=404, detail=f"Could not fetch details for token {token_id}")
        
        # Generate analyses
        tokenomics_analysis = analysis.get_investment_analysis(token_details)
        project_research = analysis.get_project_research(token_details)
        market_context = analysis.get_market_context_analysis(token_details)
        
        if not all([tokenomics_analysis, project_research, market_context]):
            raise HTTPException(status_code=500, detail="Error generating complete analysis")
        
        # Generate final report
        final_report = analysis.generate_investment_report(
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
            market_context=market_context,
            analysis_style=request.style
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 