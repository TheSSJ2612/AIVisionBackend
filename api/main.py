from dotenv import load_dotenv

# Force override any existing environment variables:
load_dotenv(dotenv_path=r"C:\Users\ShubhamTaneja\backendModels\AIVisionBackend\.env", override=True)


import os
print("GOOGLE_APPLICATION_CREDENTIALS:",
      os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


from fastapi import FastAPI
from .src.routes.users import user_router
from .src.routes.ai import ai_router

app = FastAPI()
# app.include_router(user_router)
app.include_router(ai_router)
