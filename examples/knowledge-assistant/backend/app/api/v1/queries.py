"""
Query API Endpoints

Endpoints for RAG-based question answering.
"""

from typing import Annotated
from fastapi import (
    APIRouter,
    HTTPException,
    status,
    Depends,
)
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    StreamQueryRequest,
)
from app.services.rag_service import RAGService
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


def get_rag_service() -> RAGService:
    """Dependency to get RAGService instance."""
    return RAGService()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Query",
    description="Execute a RAG query against a vector store",
)
async def execute_query(
    request: QueryRequest,
    service: Annotated[RAGService, Depends(get_rag_service)],
):
    """
    Execute a RAG query and return the response.

    This endpoint:
    - Searches the vector store for relevant documents
    - Uses the RAG agent to generate a contextual answer
    - Maintains conversation history if session_id is provided
    - Returns citations and sources

    Args:
        request: Query request with question and parameters
        service: RAG service instance

    Returns:
        Query response with answer and metadata
    """
    try:
        logger.info(f"Executing query: {request.query[:100]}...")

        response = await service.query(
            query=request.query,
            vector_store_id=request.vector_store_id,
            user_id=request.user_id,
            session_id=request.session_id,
            temperature=request.temperature,
        )

        return response

    except ValueError as e:
        logger.error(f"Invalid query request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@router.post(
    "/query/stream",
    summary="Stream Query Response",
    description="Execute a RAG query with streaming response",
)
async def stream_query(
    request: StreamQueryRequest,
    service: Annotated[RAGService, Depends(get_rag_service)],
):
    """
    Execute a RAG query with streaming response.

    This endpoint streams the response as it's being generated,
    providing a better user experience for long responses.

    Args:
        request: Stream query request
        service: RAG service instance

    Returns:
        Streaming response with answer chunks
    """
    try:
        logger.info(f"Executing streaming query: {request.query[:100]}...")

        async def generate():
            try:
                async for chunk in service.stream_query(
                    query=request.query,
                    vector_store_id=request.vector_store_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                ):
                    yield chunk
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"\n\nError: {str(e)}"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
        )

    except Exception as e:
        logger.error(f"Streaming query setup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming query failed: {str(e)}",
        )


@router.get(
    "/sessions/{session_id}",
    summary="Get Session Info",
    description="Get information about a conversation session",
)
async def get_session_info(
    session_id: str,
):
    """
    Get information about a conversation session.

    Args:
        session_id: Session identifier

    Returns:
        Session information
    """
    try:
        # This is a placeholder for future session management
        # You would typically query your session store here
        return {
            "session_id": session_id,
            "status": "active",
            "message": "Session management coming soon",
        }
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
