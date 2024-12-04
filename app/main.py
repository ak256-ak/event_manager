from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.dependencies import get_settings
from app.routers.user_routes import router as user_router


app = FastAPI(
    title="Event Manager API",
    description="An API for managing events and users.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(user_router, prefix="/api/v1")


@app.get("/")
async def read_root():
    """
    Root endpoint to confirm the service is running.
    """
    return {"message": "Welcome to the Event Manager API!"}
