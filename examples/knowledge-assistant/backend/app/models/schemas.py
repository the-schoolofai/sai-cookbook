"""
Pydantic Models for Request/Response Validation

All API request and response models with comprehensive validation.
"""

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


# Vector Store Models
# -------------------


class VectorStoreCreate(BaseModel):
    """Request model for creating a new vector store."""

    name: str = Field(
        ...,
        description="Name for the vector store",
        min_length=1,
        max_length=100,
    )
    description: str | None = Field(
        None,
        description="Optional description of the vector store",
        max_length=500,
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is valid."""
        if not v.strip():
            raise ValueError("Name cannot be empty or whitespace")
        return v.strip()


class VectorStoreResponse(BaseModel):
    """Response model for vector store operations."""

    id: str = Field(..., description="Vector store ID")
    name: str = Field(..., description="Vector store name")
    created_at: int = Field(..., description="Creation timestamp")
    file_count: int = Field(..., description="Number of files in the store")
    status: str = Field(..., description="Vector store status")
    description: str | None = Field(None, description="Vector store description")


class VectorStoreList(BaseModel):
    """Response model for listing vector stores."""

    vector_stores: list[VectorStoreResponse] = Field(
        ..., description="List of vector stores"
    )
    total: int = Field(..., description="Total number of vector stores")


# File Upload Models
# ------------------


class FileUploadResponse(BaseModel):
    """Response model for single file upload."""

    file_id: str = Field(..., description="Uploaded file ID")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    status: Literal["success", "failed"] = Field(..., description="Upload status")
    message: str | None = Field(None, description="Status message or error")


class BatchUploadRequest(BaseModel):
    """Request model for batch file upload."""

    vector_store_id: str = Field(..., description="Target vector store ID")

    @field_validator("vector_store_id")
    @classmethod
    def validate_vector_store_id(cls, v: str) -> str:
        """Validate vector store ID format."""
        if not v.startswith("vs_"):
            raise ValueError("Invalid vector store ID format")
        return v


class BatchUploadResponse(BaseModel):
    """Response model for batch upload operation."""

    total_files: int = Field(..., description="Total files processed")
    successful_uploads: int = Field(..., description="Number of successful uploads")
    failed_uploads: int = Field(..., description="Number of failed uploads")
    results: list[FileUploadResponse] = Field(
        ..., description="Individual upload results"
    )
    processing_time_seconds: float = Field(..., description="Total processing time")


# Query Models
# ------------


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(
        ...,
        description="User's question or query",
        min_length=3,
        max_length=2000,
    )
    vector_store_id: str = Field(..., description="Vector store to query")
    user_id: str | None = Field(
        None,
        description="Optional user identifier for session tracking",
        max_length=100,
    )
    session_id: str | None = Field(
        None,
        description="Optional session ID to continue conversation",
        max_length=200,
    )
    max_results: int | None = Field(
        None,
        description="Maximum number of search results to consider",
        ge=1,
        le=20,
    )
    temperature: float | None = Field(
        None,
        description="Model temperature for response generation",
        ge=0.0,
        le=2.0,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()

    @field_validator("vector_store_id")
    @classmethod
    def validate_vector_store_id(cls, v: str) -> str:
        """Validate vector store ID format."""
        if not v.startswith("vs_"):
            raise ValueError("Invalid vector store ID format")
        return v


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    answer: str = Field(..., description="Agent's response to the query")
    query: str = Field(..., description="Original query")
    vector_store_id: str = Field(..., description="Vector store used")
    session_id: str | None = Field(
        None, description="Session ID for conversation tracking"
    )
    turn_count: int | None = Field(None, description="Number of agent turns taken")
    processing_time_seconds: float = Field(..., description="Query processing time")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Additional metadata about the response",
    )


class StreamQueryRequest(BaseModel):
    """Request model for streaming RAG queries."""

    query: str = Field(
        ...,
        description="User's question or query",
        min_length=3,
        max_length=2000,
    )
    vector_store_id: str = Field(..., description="Vector store to query")
    user_id: str | None = Field(None, description="Optional user identifier")
    session_id: str | None = Field(None, description="Optional session ID")


# Search Models
# -------------


class SearchRequest(BaseModel):
    """Request model for direct vector store search."""

    query: str = Field(
        ...,
        description="Search query",
        min_length=1,
        max_length=500,
    )
    vector_store_id: str = Field(..., description="Vector store to search")
    max_results: int = Field(
        default=5,
        description="Maximum number of results",
        ge=1,
        le=20,
    )


class SearchResult(BaseModel):
    """Model for a single search result."""

    filename: str = Field(..., description="Source document filename")
    content: str = Field(..., description="Relevant content excerpt")
    score: float = Field(..., description="Relevance score")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class SearchResponse(BaseModel):
    """Response model for search operations."""

    query: str = Field(..., description="Original search query")
    results: list[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    processing_time_seconds: float = Field(..., description="Search processing time")


# Session Models
# --------------


class SessionCreate(BaseModel):
    """Request model for creating a new session."""

    user_id: str = Field(..., description="User identifier", max_length=100)
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional session metadata",
    )


class SessionResponse(BaseModel):
    """Response model for session operations."""

    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    created_at: datetime = Field(..., description="Session creation timestamp")
    message_count: int = Field(default=0, description="Number of messages in session")
    metadata: dict[str, Any] | None = Field(None, description="Session metadata")


# Error Models
# ------------


class ErrorResponse(BaseModel):
    """Standard error response model."""

    detail: str = Field(..., description="Error message")
    error_code: str | None = Field(None, description="Application-specific error code")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp",
    )


class ValidationErrorDetail(BaseModel):
    """Detailed validation error information."""

    loc: list[str | int] = Field(..., description="Error location")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ValidationErrorResponse(BaseModel):
    """Response model for validation errors."""

    detail: list[ValidationErrorDetail] = Field(..., description="Validation errors")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp",
    )


# Health Check Models
# -------------------


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall health status",
    )
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Health check timestamp",
    )
    services: dict[str, str] = Field(
        ...,
        description="Status of individual services",
    )
