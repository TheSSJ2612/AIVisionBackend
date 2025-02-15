import os
from dotenv import load_dotenv
from fastapi import FastAPI
from api.src.routes.ai import ai_router

# Force override any existing environment variables:
load_dotenv(dotenv_path="./.env", override=True)
print("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY"))
print("TAVILY_API_KEY", os.environ.get("TAVILY_API_KEY"))


# print(
#     "GOOGLE_APPLICATION_CREDENTIALS:", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
# )

app = FastAPI()
# app.include_router(user_router)
app.include_router(ai_router)
