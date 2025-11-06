"""
Vector Store API Endpoints

Endpoints for managing vector stores and file uploads.
"""

from typing import Annotated
from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    status,
    Depends,
)

from app.models.schemas import (
    VectorStoreCreate,
    VectorStoreResponse,
    VectorStoreList,
    BatchUploadResponse,
    FileUploadResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.vector_store_service import VectorStoreService
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


def get_vector_store_service() -> VectorStoreService:
    """Dependency to get VectorStoreService instance."""
    return VectorStoreService()


@router.post(
    "/vector-stores",
    response_model=VectorStoreResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Vector Store",
    description="Create a new vector store for storing and searching documents",
)
async def create_vector_store(
    data: VectorStoreCreate,
    service: Annotated[VectorStoreService, Depends(get_vector_store_service)],
):
    """
    Create a new vector store.

    Args:
        data: Vector store creation data
        service: Vector store service instance

    Returns:
        Created vector store details
    """
    try:
        return await service.create_vector_store(
            name=data.name,
            description=data.description,
        )
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create vector store: {str(e)}",
        )


@router.get(
    "/vector-stores",
    response_model=VectorStoreList,
    summary="List Vector Stores",
    description="List all available vector stores",
)
async def list_vector_stores(
    service: Annotated[VectorStoreService, Depends(get_vector_store_service)],
    limit: int = 20,
    order: str = "desc",
):
    """
    List all vector stores.

    Args:
        limit: Maximum number of stores to return
        order: Sort order (asc or desc)
        service: Vector store service instance

    Returns:
        List of vector stores
    """
    try:
        stores = await service.list_vector_stores(limit=limit, order=order)
        return VectorStoreList(
            vector_stores=stores,
            total=len(stores),
        )
    except Exception as e:
        logger.error(f"Failed to list vector stores: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list vector stores: {str(e)}",
        )


@router.get(
    "/vector-stores/{vector_store_id}",
    response_model=VectorStoreResponse,
    summary="Get Vector Store",
    description="Get details of a specific vector store",
)
async def get_vector_store(
    vector_store_id: str,
    service: Annotated[VectorStoreService, Depends(get_vector_store_service)],
):
    """
    Get vector store details.

    Args:
        vector_store_id: Vector store ID
        service: Vector store service instance

    Returns:
        Vector store details
    """
    try:
        return await service.get_vector_store(vector_store_id)
    except Exception as e:
        logger.error(f"Failed to get vector store {vector_store_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector store not found: {str(e)}",
        )


@router.delete(
    "/vector-stores/{vector_store_id}",
    summary="Delete Vector Store",
    description="Delete a vector store and all its files",
)
async def delete_vector_store(
    vector_store_id: str,
    service: Annotated[VectorStoreService, Depends(get_vector_store_service)],
):
    """
    Delete a vector store.

    Args:
        vector_store_id: Vector store ID
        service: Vector store service instance

    Returns:
        Deletion confirmation
    """
    try:
        return await service.delete_vector_store(vector_store_id)
    except Exception as e:
        logger.error(f"Failed to delete vector store {vector_store_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete vector store: {str(e)}",
        )


@router.post(
    "/vector-stores/{vector_store_id}/files",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload File",
    description="Upload a single file to a vector store",
)
async def upload_file(
    vector_store_id: str,
    file: Annotated[UploadFile, File(description="PDF file to upload")],
    service: Annotated[VectorStoreService, Depends(get_vector_store_service)],
):
    """
    Upload a single file to vector store.

    Args:
        vector_store_id: Target vector store ID
        file: File to upload
        service: Vector store service instance

    Returns:
        Upload result
    """
    try:
        return await service.upload_file(file, vector_store_id)
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}",
        )


@router.post(
    "/vector-stores/{vector_store_id}/files/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch Upload Files",
    description="Upload multiple files to a vector store",
)
async def batch_upload_files(
    vector_store_id: str,
    files: Annotated[list[UploadFile], File(description="PDF files to upload")],
    service: Annotated[VectorStoreService, Depends(get_vector_store_service)],
):
    """
    Upload multiple files to vector store.

    Args:
        vector_store_id: Target vector store ID
        files: List of files to upload
        service: Vector store service instance

    Returns:
        Batch upload results
    """
    try:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided",
            )

        return await service.batch_upload_files(files, vector_store_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to batch upload files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch upload files: {str(e)}",
        )


@router.post(
    "/vector-stores/search",
    response_model=SearchResponse,
    summary="Search Vector Store",
    description="Search for relevant documents in a vector store",
)
async def search_vector_store(
    request: SearchRequest,
    service: Annotated[VectorStoreService, Depends(get_vector_store_service)],
):
    """
    Search vector store for relevant documents.

    Args:
        request: Search request with query and parameters
        service: Vector store service instance

    Returns:
        Search results
    """
    import time

    start_time = time.time()

    try:
        results = await service.search_vector_store(
            query=request.query,
            vector_store_id=request.vector_store_id,
            max_results=request.max_results,
        )

        processing_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_seconds=processing_time,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )
