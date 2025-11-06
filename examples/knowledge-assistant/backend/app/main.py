"""
Knowledge Assistant Backend - Main Application

FastAPI application for RAG-based PDF knowledge assistant.

Author: SAI Engineering Team
Date: November 2025
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

# from fastapi import Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# from fastapi.responses import FileResponse, Response, StreamingResponse

from app.api.v1 import vector_stores, queries, health
from app.config.settings import get_settings
from app.core.logging import setup_logging

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    settings = get_settings()
    logger.info("Starting Knowledge Assistant Backend...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"API Version: {settings.API_VERSION}")

    yield

    logger.info("Shutting down Knowledge Assistant Backend...")


# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Assistant API",
    description="RAG system for PDF document question answering",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred",
            "type": "internal_error",
        },
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(vector_stores.router, prefix="/api/v1", tags=["Vector Stores"])
app.include_router(queries.router, prefix="/api/v1", tags=["Queries"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Knowledge Assistant API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/api/docs",
    }


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=settings.ENVIRONMENT == "development",
#         log_level="info",
#     )

# uv run --active fastapi dev app/main.py
