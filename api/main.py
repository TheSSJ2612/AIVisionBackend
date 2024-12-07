from fastapi import FastAPI
from .src.routes.users import user_router
from .src.routes.ai import ai_router

app = FastAPI()
# app.include_router(user_router)
app.include_router(ai_router)
