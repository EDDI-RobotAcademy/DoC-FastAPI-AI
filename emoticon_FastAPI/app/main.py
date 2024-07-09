import os

import aiomysql
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from emoticon_FastAPI.async_db.database import getMySqlPool, createTableIfNeccessary
from emoticon_FastAPI.lgbm_analysis.controller.lgbm_controller import lgbmAnalysisRouter

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

import warnings

warnings.filterwarnings("ignore", category=aiomysql.Warning)

async def lifespan(app: FastAPI):
    # Startup
    app.state.dbPool = await getMySqlPool()
    await createTableIfNeccessary(app.state.dbPool)

    yield

    # Shutdown
    app.state.dbPool.close()
    await app.state.dbPool.wait_closed()

app.include_router(lgbmAnalysisRouter)
load_dotenv()

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.24", port=33333) #192.168.0.24

    # 위에 각자 ip 주소로 변환해서 사용