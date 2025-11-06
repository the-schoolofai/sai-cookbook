"""
Vector Store Service

Business logic for vector store operations.
"""

import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from fastapi import UploadFile

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.models.schemas import (
    VectorStoreResponse,
    FileUploadResponse,
    BatchUploadResponse,
    SearchResult,
)

logger = get_logger(__name__)


class VectorStoreService:
    """Service for managing vector store operations."""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        Path(self.settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    async def create_vector_store(
        self,
        name: str,
        description: str | None = None,
    ) -> VectorStoreResponse:
        """
        Create a new vector store.

        Args:
            name: Name for the vector store
            description: Optional description

        Returns:
            VectorStoreResponse with store details
        """
        try:
            logger.info(f"Creating vector store: {name}")

            vector_store = self.client.vector_stores.create(
                name=name,
                metadata={"description": description} if description else {},
            )

            response = VectorStoreResponse(
                id=vector_store.id,
                name=vector_store.name,
                created_at=vector_store.created_at,
                file_count=vector_store.file_counts.completed,
                status="ready",
                description=description,
            )

            logger.info(f"Vector store created successfully: {vector_store.id}")
            return response

        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    async def get_vector_store(self, vector_store_id: str) -> VectorStoreResponse:
        """
        Get vector store details.

        Args:
            vector_store_id: Vector store ID

        Returns:
            VectorStoreResponse with store details
        """
        try:
            vector_store = self.client.vector_stores.retrieve(vector_store_id)

            return VectorStoreResponse(
                id=vector_store.id,
                name=vector_store.name,
                created_at=vector_store.created_at,
                file_count=vector_store.file_counts.completed,
                status=vector_store.status,
                description=vector_store.metadata.get("description"),
            )

        except Exception as e:
            logger.error(f"Failed to retrieve vector store {vector_store_id}: {e}")
            raise

    async def list_vector_stores(
        self,
        limit: int = 20,
        order: str = "desc",
    ) -> list[VectorStoreResponse]:
        """
        List all vector stores.

        Args:
            limit: Maximum number of stores to return
            order: Sort order (asc or desc)

        Returns:
            List of VectorStoreResponse objects
        """
        try:
            vector_stores = self.client.vector_stores.list(
                limit=limit,
                order=order,
            )

            return [
                VectorStoreResponse(
                    id=vs.id,
                    name=vs.name,
                    created_at=vs.created_at,
                    file_count=vs.file_counts.completed,
                    status=vs.status,
                    description=vs.metadata.get("description"),
                )
                for vs in vector_stores.data
            ]

        except Exception as e:
            logger.error(f"Failed to list vector stores: {e}")
            raise

    async def delete_vector_store(self, vector_store_id: str) -> dict[str, Any]:
        """
        Delete a vector store.

        Args:
            vector_store_id: Vector store ID

        Returns:
            Deletion confirmation
        """
        try:
            logger.info(f"Deleting vector store: {vector_store_id}")

            result = self.client.vector_stores.delete(vector_store_id)

            logger.info(f"Vector store deleted successfully: {vector_store_id}")
            return {
                "id": result.id,
                "deleted": result.deleted,
                "message": "Vector store deleted successfully",
            }

        except Exception as e:
            logger.error(f"Failed to delete vector store {vector_store_id}: {e}")
            raise

    async def upload_file(
        self,
        file: UploadFile,
        vector_store_id: str,
    ) -> FileUploadResponse:
        """
        Upload a single file to vector store.

        Args:
            file: File to upload
            vector_store_id: Target vector store ID

        Returns:
            FileUploadResponse with upload details
        """
        start_time = time.time()

        try:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in self.settings.ALLOWED_EXTENSIONS:
                return FileUploadResponse(
                    file_id="",
                    filename=file.filename,
                    size_bytes=0,
                    status="failed",
                    message=f"File type {file_ext} not allowed",
                )

            # Read file content
            content = await file.read()
            file_size = len(content)

            # Validate file size
            if file_size > self.settings.max_file_size_bytes:
                return FileUploadResponse(
                    file_id="",
                    filename=file.filename,
                    size_bytes=file_size,
                    status="failed",
                    message=f"File size exceeds {self.settings.MAX_FILE_SIZE_MB}MB limit",
                )

            # Upload to OpenAI
            file_response = self.client.files.create(
                file=(file.filename, content),
                purpose="assistants",
            )

            # Attach to vector store
            self.client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_response.id,
            )

            logger.info(
                f"File uploaded successfully: {file.filename} "
                f"({file_size} bytes) in {time.time() - start_time:.2f}s"
            )

            return FileUploadResponse(
                file_id=file_response.id,
                filename=file.filename,
                size_bytes=file_size,
                status="success",
                message="File uploaded successfully",
            )

        except Exception as e:
            logger.error(f"Failed to upload file {file.filename}: {e}")
            return FileUploadResponse(
                file_id="",
                filename=file.filename,
                size_bytes=0,
                status="failed",
                message=str(e),
            )

    async def batch_upload_files(
        self,
        files: list[UploadFile],
        vector_store_id: str,
    ) -> BatchUploadResponse:
        """
        Upload multiple files in parallel.

        Args:
            files: List of files to upload
            vector_store_id: Target vector store ID

        Returns:
            BatchUploadResponse with upload statistics
        """
        start_time = time.time()
        results = []

        logger.info(f"Starting batch upload of {len(files)} files")

        # Upload files sequentially (FastAPI handles async)
        for file in files:
            result = await self.upload_file(file, vector_store_id)
            results.append(result)

        successful = sum(1 for r in results if r.status == "success")
        failed = len(results) - successful
        processing_time = time.time() - start_time

        logger.info(
            f"Batch upload completed: {successful}/{len(files)} successful "
            f"in {processing_time:.2f}s"
        )

        return BatchUploadResponse(
            total_files=len(files),
            successful_uploads=successful,
            failed_uploads=failed,
            results=results,
            processing_time_seconds=processing_time,
        )

    async def search_vector_store(
        self,
        query: str,
        vector_store_id: str,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """
        Search vector store for relevant documents.

        Args:
            query: Search query
            vector_store_id: Vector store to search
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"Searching vector store {vector_store_id} for: {query}")

            search_results = self.client.vector_stores.search(
                vector_store_id=vector_store_id,
                query=query,
            )

            results = []
            for result in search_results.data[:max_results]:
                content = result.content[0].text if result.content else ""

                results.append(
                    SearchResult(
                        filename=result.filename,
                        content=content[:500] + "..."
                        if len(content) > 500
                        else content,
                        score=result.score,
                        metadata={"full_content_length": len(content)},
                    )
                )

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
