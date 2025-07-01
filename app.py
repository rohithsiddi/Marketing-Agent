from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio
from MarketingAgent import marketing_agent

class QueryRequest(BaseModel):
    message: str
    success_criteria: str = ""

app = FastAPI()

agent = marketing_agent()

@app.on_event("startup")
async def startup_event():
    await agent.setup()

@app.post("/run-agent")
async def run_agent(request: QueryRequest):
    result = await agent.run_superstep(request.message, request.success_criteria, [])
    return {"result": result}


