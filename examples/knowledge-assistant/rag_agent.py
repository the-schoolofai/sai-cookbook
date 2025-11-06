"""
RAG Agent

This script provides a complete implementation of a RAG system
using OpenAI's File Search and Agents Python SDK.

Features:
- Vector store management
- File upload with parallel processing
- RAG agent with conversation history
- Interactive chat interface
- Performance monitoring
- Comprehensive error handling

Usage:
    python rag_agent.py

Requirements:
    uv add openai openai-agents tqdm

Author: SAI Engineering Team
Date: November 2025
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# OpenAI Python SDK
from openai import OpenAI

# OpenAI Agents Python SDK
from agents import (
    Agent,
    Runner,
    RunConfig,
    ModelSettings,
    FileSearchTool,
    SQLiteSession,
)

# Utilities
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field


# CONFIGURATION
# -------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Application configuration with environment variable support."""

    # API Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # File Storage
    pdf_directory: str = "data/openai_blog_pdfs"
    vector_store_name: str = "openai_blog_store"

    # Model Settings
    model_name: str = "gpt-4.1"
    temperature: float = 0.3
    max_tokens: int = 2048

    # Processing Settings
    max_workers: int = 10
    max_turns: int = 10

    # Session Settings
    session_prefix: str = "rag_session"

    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in environment variables or .env file."
            )

        if not Path(self.pdf_directory).exists():
            logger.warning(
                f"PDF directory '{self.pdf_directory}' not found. Creating it..."
            )
            Path(self.pdf_directory).mkdir(parents=True, exist_ok=True)


# DATA MODELS
# -----------


class RAGQuery(BaseModel):
    """Input model for RAG queries with validation."""

    query: str = Field(
        ..., description="User's question or query", min_length=3, max_length=1000
    )
    user_id: str | None = Field(
        None, description="Optional user identifier for session tracking"
    )
    session_id: str | None = Field(
        None, description="Optional session ID to continue existing conversation"
    )


class RAGResponse(BaseModel):
    """Response model from RAG agent with metadata."""

    answer: str = Field(..., description="The agent's response to the query")
    session_id: str | None = Field(
        None, description="Session ID for conversation tracking"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )
    turn_count: int | None = Field(None, description="Number of agent turns taken")


# VECTOR STORE MANAGER
# --------------------


class VectorStoreManager:
    """Manages OpenAI vector store operations."""

    def __init__(self, client: OpenAI):
        self.client = client
        self.logger = logging.getLogger(f"{__name__}.VectorStoreManager")

    def create_vector_store(self, store_name: str) -> dict[str, Any]:
        """
        Create a new vector store.

        Args:
            store_name: Name for the vector store

        Returns:
            Dictionary containing vector store details
        """
        try:
            self.logger.info(f"Creating vector store: {store_name}")
            vector_store = self.client.vector_stores.create(name=store_name)

            details = {
                "id": vector_store.id,
                "name": vector_store.name,
                "created_at": vector_store.created_at,
                "file_count": vector_store.file_counts.completed,
                "status": "ready",
            }

            self.logger.info(f"[SUCCESS] Vector store created: {details['id']}")
            return details

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to create vector store: {e}")
            raise

    def upload_single_file(
        self, file_path: str, vector_store_id: str
    ) -> dict[str, str]:
        """
        Upload a single file to the vector store.

        Args:
            file_path: Path to the file
            vector_store_id: Target vector store ID

        Returns:
            Upload result dictionary
        """
        file_name = os.path.basename(file_path)

        try:
            # Upload file to OpenAI
            with open(file_path, "rb") as f:
                file_response = self.client.files.create(file=f, purpose="assistants")

            # Attach file to vector store
            self.client.vector_stores.files.create(
                vector_store_id=vector_store_id, file_id=file_response.id
            )

            return {"file": file_name, "file_id": file_response.id, "status": "success"}

        except Exception as e:
            self.logger.error(f"Failed to upload {file_name}: {str(e)}")
            return {"file": file_name, "status": "failed", "error": str(e)}

    def upload_files_parallel(
        self, pdf_directory: str, vector_store_id: str, max_workers: int = 10
    ) -> dict[str, Any]:
        """
        Upload multiple files in parallel for better performance.

        Args:
            pdf_directory: Directory containing PDF files
            vector_store_id: Target vector store ID
            max_workers: Maximum concurrent uploads

        Returns:
            Statistics dictionary with upload results
        """
        pdf_files = [
            os.path.join(pdf_directory, f)
            for f in os.listdir(pdf_directory)
            if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            self.logger.warning(f"No PDF files found in {pdf_directory}")
            return {
                "total_files": 0,
                "successful_uploads": 0,
                "failed_uploads": 0,
                "errors": [],
            }

        stats = {
            "total_files": len(pdf_files),
            "successful_uploads": 0,
            "failed_uploads": 0,
            "errors": [],
        }

        self.logger.info(
            f"[UPLOAD] Uploading {len(pdf_files)} files in parallel (max_workers={max_workers})"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.upload_single_file, file_path, vector_store_id
                ): file_path
                for file_path in pdf_files
            }

            for future in tqdm(futures, total=len(pdf_files), desc="Uploading PDFs"):
                result = future.result()

                if result["status"] == "success":
                    stats["successful_uploads"] += 1
                else:
                    stats["failed_uploads"] += 1
                    stats["errors"].append(result)

        self.logger.info(
            f"[SUCCESS] Upload complete: {stats['successful_uploads']}/{stats['total_files']} successful"
        )

        if stats["errors"]:
            self.logger.warning(
                f"[WARNING] {stats['failed_uploads']} files failed to upload"
            )

        return stats

    def search_vector_store(
        self, query: str, vector_store_id: str, max_results: int = 5
    ) -> None:
        """
        Search the vector store and display results.

        Args:
            query: Search query in natural language
            vector_store_id: ID of the vector store to search
            max_results: Maximum number of results to display
        """
        self.logger.info(f"[SEARCH] Searching for: '{query}'")

        try:
            search_results = self.client.vector_stores.search(
                vector_store_id=vector_store_id, query=query
            )

            print("\n" + "=" * 80)
            print(f"Search Results for: '{query}'")
            print("=" * 80)

            if not search_results.data:
                print("No results found.")
                return

            for idx, result in enumerate(search_results.data[:max_results], 1):
                content_length = len(result.content[0].text) if result.content else 0

                print(f"\n[{idx}] {result.filename}")
                print(f"    Relevance Score: {result.score:.4f}")
                print(f"    Content Length:  {content_length} characters")

                # Display preview of content
                if result.content and content_length > 0:
                    preview = result.content[0].text[:200].replace("\n", " ")
                    print(f"    Preview: {preview}...")

            print("\n" + "=" * 80)

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise


# RAG AGENT
# ---------


class RAGAgent:
    """RAG agent with file search capabilities."""

    def __init__(self, config: Config, vector_store_id: str):
        self.config = config
        self.vector_store_id = vector_store_id
        self.logger = logging.getLogger(f"{__name__}.RAGAgent")

        # Initialize File Search Tool
        self.file_search_tool = FileSearchTool(
            vector_store_ids=[vector_store_id], max_num_results=5
        )

        # Create Agent
        self.agent = Agent(
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
            model=config.model_name,
            tools=[self.file_search_tool],
            model_settings=ModelSettings(
                temperature=config.temperature,
                top_p=0.9,
                max_tokens=config.max_tokens,
            ),
        )

        self.logger.info("[INIT] RAG Agent initialized successfully")

    async def run(
        self,
        query: str,
        user_id: str | None = None,
        session: SQLiteSession | None = None,
        verbose: bool = True,
    ) -> RAGResponse:
        """
        Run the RAG agent with a user query.

        Args:
            query: User's question or input
            user_id: Optional user identifier for tracking
            session: Optional SQLiteSession for maintaining conversation history
            verbose: Whether to print detailed information

        Returns:
            RAGResponse containing the agent's answer and metadata
        """
        # Validate input
        rag_query = RAGQuery(query=query, user_id=user_id)

        try:
            # Create or use provided session
            if session is None and user_id:
                session = SQLiteSession(f"{self.config.session_prefix}_{user_id}")

            if verbose:
                self.logger.info(f"[RUN] Running agent for query: '{query[:50]}...'")

            # Run the agent with configuration
            result = await Runner.run(
                self.agent,
                input=query,
                session=session,
                max_turns=self.config.max_turns,
                run_config=RunConfig(
                    trace_metadata={
                        "__trace_source__": "rag-agent",
                        "user_id": user_id or "anonymous",
                        "query_timestamp": datetime.now().isoformat(),
                    }
                ),
            )

            # Extract final output
            answer = result.final_output_as(str)

            if verbose:
                self.logger.info("[SUCCESS] Agent completed successfully")

            return RAGResponse(
                answer=answer,
                session_id=session.session_id if session else None,
                turn_count=getattr(result, "total_turns", None),
            )

        except ValueError as e:
            self.logger.error(f"[ERROR] Invalid query: {e}")
            raise

        except Exception as e:
            self.logger.error(f"[ERROR] Error running RAG agent: {e}")
            return RAGResponse(
                answer="I apologize, but I encountered an error processing your request. Please try again.",
                session_id=session.session_id if session else None,
            )


# INTERACTIVE CHAT
# ----------------


async def interactive_chat(rag_agent: RAGAgent, user_id: str = "demo_user"):
    """
    Run an interactive chat session with the RAG agent.

    Args:
        rag_agent: Initialized RAG agent instance
        user_id: Identifier for the user session
    """
    print("\n" + "=" * 80)
    print("PDF Knowledge Assistant - Interactive Chat")
    print("=" * 80)
    print("\nWelcome! Ask me anything about the documents in the knowledge base.")
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - Type 'exit', 'quit', or 'bye' to end the conversation")
    print("  - Type 'clear' to start a new conversation")
    print("  - Type 'help' to see available commands")
    print("\n" + "=" * 80 + "\n")

    # Create session for this conversation
    session = SQLiteSession(f"{rag_agent.config.session_prefix}_{user_id}")
    turn_count = 0

    while True:
        try:
            # Get user input
            user_input = input("\n[You]: ").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nThank you for chatting! Goodbye!\n")
                break

            if user_input.lower() == "clear":
                session = SQLiteSession(
                    f"{rag_agent.config.session_prefix}_{user_id}_new"
                )
                turn_count = 0
                print("\nConversation cleared. Starting fresh!\n")
                continue

            if user_input.lower() == "help":
                print("\nAvailable Commands:")
                print("  - exit/quit/bye: End the conversation")
                print("  - clear: Start a new conversation")
                print("  - help: Show this help message")
                continue

            # Run the agent
            turn_count += 1
            print(f"\n[Assistant] Turn {turn_count}: ", end="", flush=True)

            response = await rag_agent.run(
                query=user_input, user_id=user_id, session=session, verbose=False
            )

            print(response.answer)

        except KeyboardInterrupt:
            print("\n\n[WARNING] Conversation interrupted. Goodbye!\n")
            break

        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print(f"\n[ERROR] Error: {e}")
            print("Please try again.\n")


# MAIN APPLICATION
# ----------------


async def main():
    """Main application entry point."""

    print("\n" + "=" * 80)
    print("RAG Agent")
    print("=" * 80 + "\n")

    # Initialize configuration
    config = Config()
    config.validate()

    # Initialize OpenAI client
    client = OpenAI(api_key=config.openai_api_key)
    logger.info("[SUCCESS] OpenAI client initialized")

    # Initialize Vector Store Manager
    vs_manager = VectorStoreManager(client)

    # Ask user what they want to do
    print("\nOptions:")
    print("1. Create new vector store and upload files")
    print("2. Use existing vector store")
    print("3. Search vector store directly")
    print("4. Start interactive chat")

    choice = input("\nEnter your choice (1-4): ").strip()

    vector_store_id = None

    if choice == "1":
        # Create vector store
        print("\n[STEP 1] Creating vector store...")
        vector_store_details = vs_manager.create_vector_store(config.vector_store_name)
        vector_store_id = vector_store_details["id"]

        print("\n[SUCCESS] Vector Store Created:")
        print(f"   ID: {vector_store_id}")
        print(f"   Name: {vector_store_details['name']}")

        # Upload files
        print(f"\n[STEP 2] Uploading files from '{config.pdf_directory}'...")
        upload_stats = vs_manager.upload_files_parallel(
            pdf_directory=config.pdf_directory,
            vector_store_id=vector_store_id,
            max_workers=config.max_workers,
        )

        print("\n[SUCCESS] Upload Statistics:")
        print(f"   Total: {upload_stats['total_files']}")
        print(f"   Successful: {upload_stats['successful_uploads']}")
        print(f"   Failed: {upload_stats['failed_uploads']}")

    elif choice == "2":
        vector_store_id = input("\nEnter vector store ID: ").strip()

    elif choice == "3":
        vector_store_id = input("\nEnter vector store ID: ").strip()
        query = input("Enter search query: ").strip()
        vs_manager.search_vector_store(query, vector_store_id)
        return

    elif choice == "4":
        vector_store_id = input("\nEnter vector store ID: ").strip()

    else:
        print("Invalid choice. Exiting.")
        return

    # Initialize RAG Agent
    print("\n[INIT] Initializing RAG Agent...")
    rag_agent = RAGAgent(config=config, vector_store_id=vector_store_id)

    # Ask what to do next
    print("\nWhat would you like to do?")
    print("1. Test with single query")
    print("2. Start interactive chat")

    next_choice = input("\nEnter your choice (1-2): ").strip()

    if next_choice == "1":
        query = input("\nEnter your query: ").strip()
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}\n")

        response = await rag_agent.run(query=query, verbose=True)

        print(f"\n{'=' * 80}")
        print("Response:")
        print(f"{'=' * 80}")
        print(response.answer)
        print(f"\n{'=' * 80}")

    elif next_choice == "2":
        await interactive_chat(rag_agent)

    print("\n[SUCCESS] Application completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[WARNING] Application interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n[ERROR] Fatal error: {e}")
