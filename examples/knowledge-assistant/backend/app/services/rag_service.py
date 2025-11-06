"""
RAG Service

Business logic for RAG query operations using OpenAI Agents SDK.
"""

import time
from datetime import datetime
from typing import AsyncGenerator

from openai import OpenAI
from agents import (
    Agent,
    Runner,
    RunConfig,
    ModelSettings,
    FileSearchTool,
    SQLiteSession,
)

from dotenv import load_dotenv

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.models.schemas import QueryResponse

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


class RAGService:
    """Service for RAG query operations."""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
        self._agents = {}  # Cache agents by vector_store_id

    def _get_or_create_agent(self, vector_store_id: str) -> Agent:
        """
        Get cached agent or create new one for vector store.

        Args:
            vector_store_id: Vector store ID

        Returns:
            Agent instance configured for the vector store
        """
        if vector_store_id in self._agents:
            return self._agents[vector_store_id]

        # Create file search tool
        file_search_tool = FileSearchTool(
            vector_store_ids=[vector_store_id],
            max_num_results=self.settings.MAX_NUM_RESULTS,
        )

        # Create agent
        agent = Agent(
            name="PDF Knowledge Assistant",
            instructions="""You are an intelligent assistant specializing in answering questions 
            based on PDF documents using advanced file search capabilities.
            
            ## Core Responsibilities:
            - Search and retrieve information from uploaded PDF documents
            - Synthesize information from multiple sources when needed
            - Provide accurate, contextual answers grounded in source material
            - Cite sources and reference specific documents
            
            ## Guidelines:
            
            ### Always:
            - Use the file search tool before answering questions
            - Base answers strictly on information found in documents
            - Cite sources with document names when providing information
            - Provide direct quotes for important claims
            - Maintain a clear, professional, helpful tone
            - Break down complex information into digestible explanations
            - Offer to search for related information if initial answer seems incomplete
            
            ### Never:
            - Make assumptions beyond what's in the documents
            - Speculate or provide information not in source material
            - Mix general knowledge with document-specific info without clear distinction
            - Claim certainty when information is ambiguous or missing
            
            ## Response Format:
            1. Provide direct answer to the question
            2. Include relevant context from source documents
            3. Cite source documents clearly
            4. Acknowledge if information is not found in the knowledge base
            """,
            model=self.settings.OPENAI_MODEL,
            tools=[file_search_tool],
            model_settings=ModelSettings(
                temperature=self.settings.OPENAI_TEMPERATURE,
                top_p=0.9,
                max_tokens=self.settings.OPENAI_MAX_TOKENS,
            ),
        )

        # Cache agent
        self._agents[vector_store_id] = agent
        logger.info(f"Created new agent for vector store: {vector_store_id}")

        return agent

    def _get_session(
        self, user_id: str | None, session_id: str | None
    ) -> SQLiteSession | None:
        """
        Get or create session for conversation history.

        Args:
            user_id: Optional user identifier
            session_id: Optional existing session ID

        Returns:
            SQLiteSession instance or None
        """
        if session_id:
            return SQLiteSession(session_id)
        elif user_id:
            return SQLiteSession(f"{self.settings.SESSION_PREFIX}_{user_id}")
        return None

    async def query(
        self,
        query: str,
        vector_store_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        temperature: float | None = None,
    ) -> QueryResponse:
        """
        Execute RAG query and return response.

        Args:
            query: User's question
            vector_store_id: Vector store to query
            user_id: Optional user identifier
            session_id: Optional session ID
            temperature: Optional temperature override

        Returns:
            QueryResponse with answer and metadata
        """
        start_time = time.time()

        try:
            logger.info(f"Processing query: {query[:100]}...")

            # Get agent for vector store
            agent = self._get_or_create_agent(vector_store_id)

            # Override temperature if provided
            if temperature is not None:
                agent.model_settings.temperature = temperature

            # Get session for conversation history
            session = self._get_session(user_id, session_id)

            # Run agent
            result = await Runner.run(
                agent,
                input=query,
                session=session,
                max_turns=self.settings.MAX_TURNS,
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "knowledge-assistant-api",
                        "user_id": user_id or "anonymous",
                        "query_timestamp": datetime.now().isoformat(),
                        "vector_store_id": vector_store_id,
                    }
                ),
            )

            # Extract answer
            answer = result.final_output_as(str)

            processing_time = time.time() - start_time

            logger.info(f"Query processed successfully in {processing_time:.2f}s")

            return QueryResponse(
                answer=answer,
                query=query,
                vector_store_id=vector_store_id,
                session_id=session.session_id if session else None,
                turn_count=getattr(result, "total_turns", None),
                processing_time_seconds=processing_time,
                timestamp=datetime.now(),
                metadata={
                    "model": self.settings.OPENAI_MODEL,
                    "temperature": agent.model_settings.temperature,
                },
            )

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            processing_time = time.time() - start_time

            return QueryResponse(
                answer="I apologize, but I encountered an error processing your request. Please try again.",
                query=query,
                vector_store_id=vector_store_id,
                session_id=session_id,
                processing_time_seconds=processing_time,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
            )

    async def stream_query(
        self,
        query: str,
        vector_store_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Execute RAG query with streaming response.

        Args:
            query: User's question
            vector_store_id: Vector store to query
            user_id: Optional user identifier
            session_id: Optional session ID

        Yields:
            Chunks of the response as they're generated
        """
        try:
            logger.info(f"Processing streaming query: {query[:100]}...")

            # Get agent for vector store
            agent = self._get_or_create_agent(vector_store_id)

            # Get session for conversation history
            session = self._get_session(user_id, session_id)

            # Run agent with streaming
            async for chunk in Runner.stream(
                agent,
                input=query,
                session=session,
                max_turns=self.settings.MAX_TURNS,
            ):
                if hasattr(chunk, "text"):
                    yield chunk.text

            logger.info("Streaming query completed")

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"Error: {str(e)}"
