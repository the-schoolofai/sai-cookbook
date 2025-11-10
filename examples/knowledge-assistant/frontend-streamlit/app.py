"""
Knowledge Assistant - Streamlit Frontend

Interactive web interface for the Knowledge Assistant RAG system.

Author: SAI Engineering Team
Date: November 2025
"""

import streamlit as st
import requests
import time
from datetime import datetime


# Page Configuration
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Configuration
class Config:
    """Application configuration."""

    API_BASE_URL = "http://localhost:8000/api/v1"
    TIMEOUT = 30
    MAX_FILE_SIZE_MB = 50


# API Client
class KnowledgeAssistantAPI:
    """Client for Knowledge Assistant API."""

    def __init__(self, base_url: str = Config.API_BASE_URL):
        self.base_url = base_url
        self.timeout = Config.TIMEOUT

    def health_check(self) -> dict:
        """Check API health status."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def list_vector_stores(self, limit: int = 20) -> dict:
        """List all vector stores."""
        response = requests.get(
            f"{self.base_url}/vector-stores",
            params={"limit": limit},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def create_vector_store(self, name: str, description: str = None) -> dict:
        """Create a new vector store."""
        response = requests.post(
            f"{self.base_url}/vector-stores",
            json={"name": name, "description": description},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_vector_store(self, vector_store_id: str) -> dict:
        """Get vector store details."""
        response = requests.get(
            f"{self.base_url}/vector-stores/{vector_store_id}", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def delete_vector_store(self, vector_store_id: str) -> dict:
        """Delete a vector store."""
        response = requests.delete(
            f"{self.base_url}/vector-stores/{vector_store_id}", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def upload_file(self, vector_store_id: str, file) -> dict:
        """Upload a single file."""
        files = {"file": file}
        response = requests.post(
            f"{self.base_url}/vector-stores/{vector_store_id}/files",
            files=files,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def batch_upload_files(self, vector_store_id: str, files: list) -> dict:
        """Upload multiple files."""
        files_data = [("files", file) for file in files]
        response = requests.post(
            f"{self.base_url}/vector-stores/{vector_store_id}/files/batch",
            files=files_data,
            timeout=self.timeout * 2,  # Longer timeout for batch
        )
        response.raise_for_status()
        return response.json()

    def search(self, query: str, vector_store_id: str, max_results: int = 5) -> dict:
        """Search vector store."""
        response = requests.post(
            f"{self.base_url}/vector-stores/search",
            json={
                "query": query,
                "vector_store_id": vector_store_id,
                "max_results": max_results,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def query(
        self,
        query: str,
        vector_store_id: str,
        user_id: str = None,
        session_id: str = None,
        temperature: float = None,
    ) -> dict:
        """Execute RAG query."""
        payload = {
            "query": query,
            "vector_store_id": vector_store_id,
        }
        if user_id:
            payload["user_id"] = user_id
        if session_id:
            payload["session_id"] = session_id
        if temperature is not None:
            payload["temperature"] = temperature

        response = requests.post(
            f"{self.base_url}/query", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def stream_query(
        self,
        query: str,
        vector_store_id: str,
        user_id: str = None,
        session_id: str = None,
    ):
        """Stream RAG query response."""
        payload = {
            "query": query,
            "vector_store_id": vector_store_id,
        }
        if user_id:
            payload["user_id"] = user_id
        if session_id:
            payload["session_id"] = session_id

        response = requests.post(
            f"{self.base_url}/query/stream",
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response


# Initialize API client
api = KnowledgeAssistantAPI()


# Session State Initialization
def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_vector_store" not in st.session_state:
        st.session_state.current_vector_store = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{int(time.time())}"


# Helper Functions
def display_health_status():
    """Display API health status in sidebar."""
    health = api.health_check()

    if health.get("status") == "healthy":
        st.sidebar.success("API: Healthy")
    elif health.get("status") == "degraded":
        st.sidebar.warning("API: Degraded")
    else:
        st.sidebar.error(f"API: {health.get('error', 'Unhealthy')}")


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp


def display_metric_card(label: str, value: str):
    """Display a metric card."""
    st.markdown(
        f"""
    <div class="metric-card">
        <strong>{label}</strong><br>
        {value}
    </div>
    """,
        unsafe_allow_html=True,
    )


# Page: Vector Store Management
def page_vector_stores():
    """Vector store management page."""
    st.markdown(
        '<div class="main-header"> Vector Store Management</div>',
        unsafe_allow_html=True,
    )

    # Create new vector store
    with st.expander("Create New Vector Store", expanded=False):
        with st.form("create_vector_store"):
            name = st.text_input("Name *", placeholder="e.g., technical_docs")
            description = st.text_area(
                "Description", placeholder="Optional description"
            )

            if st.form_submit_button("Create Vector Store", type="primary"):
                if not name:
                    st.error("Name is required")
                else:
                    try:
                        with st.spinner("Creating vector store..."):
                            result = api.create_vector_store(name, description)

                        st.markdown(
                            f"""
                        <div class="success-box">
                            <strong>Vector Store Created!</strong><br>
                            ID: {result["id"]}<br>
                            Name: {result["name"]}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.markdown(
                            f"""
                        <div class="error-box">
                            <strong>Error:</strong> {str(e)}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

    st.divider()

    # List existing vector stores
    st.subheader("Existing Vector Stores")

    try:
        with st.spinner("Loading vector stores..."):
            stores = api.list_vector_stores()

        if stores["total"] == 0:
            st.info("No vector stores found. Create one to get started!")
        else:
            for store in stores["vector_stores"]:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                    with col1:
                        st.markdown(f"**{store['name']}**")
                        if store.get("description"):
                            st.caption(store["description"])

                    with col2:
                        st.metric("Files", store["file_count"])

                    with col3:
                        status_color = "🟢" if store["status"] == "ready" else "🟡"
                        st.write(f"{status_color} {store['status'].title()}")

                    with col4:
                        if st.button(
                            "Select",
                            key=f"select_{store['id']}",
                            use_container_width=True,
                        ):
                            st.session_state.current_vector_store = store["id"]
                            st.success(f"Selected: {store['name']}")
                            time.sleep(1)
                            st.rerun()

                    # Expandable details
                    with st.expander("Details & Actions"):
                        col1, col2 = st.columns(2)

                        with col1:
                            display_metric_card("Vector Store ID", store["id"])
                            display_metric_card(
                                "Created",
                                datetime.fromtimestamp(store["created_at"]).strftime(
                                    "%Y-%m-%d %H:%M"
                                ),
                            )

                        with col2:
                            if st.button(
                                "Delete",
                                key=f"delete_{store['id']}",
                                type="secondary",
                            ):
                                try:
                                    with st.spinner("Deleting..."):
                                        api.delete_vector_store(store["id"])
                                    st.success("Vector store deleted!")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                    st.divider()

    except Exception as e:
        st.error(f"Error loading vector stores: {str(e)}")


# Page: File Upload
def page_file_upload():
    """File upload page."""
    st.markdown(
        '<div class="main-header"> Upload Documents</div>', unsafe_allow_html=True
    )

    # Check if vector store is selected
    if not st.session_state.current_vector_store:
        st.warning("Please select a vector store first from the 'Vector Stores' page.")
        return

    # Display selected vector store
    try:
        store = api.get_vector_store(st.session_state.current_vector_store)
        st.markdown(
            f"""
        <div class="info-box">
            <strong>Current Vector Store:</strong> {store["name"]}<br>
            <strong>Files:</strong> {store["file_count"]}
        </div>
        """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return

    st.divider()

    # Upload options
    upload_mode = st.radio(
        "Upload Mode", ["Single File", "Multiple Files (Batch)"], horizontal=True
    )

    if upload_mode == "Single File":
        st.subheader("Upload Single File")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help=f"Maximum file size: {Config.MAX_FILE_SIZE_MB}MB",
        )

        if uploaded_file:
            # Display file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"{uploaded_file.name} ({file_size_mb:.2f} MB)")

            if st.button("Upload File", type="primary"):
                if file_size_mb > Config.MAX_FILE_SIZE_MB:
                    st.error(f"File size exceeds {Config.MAX_FILE_SIZE_MB}MB limit")
                else:
                    try:
                        with st.spinner("Uploading..."):
                            result = api.upload_file(
                                st.session_state.current_vector_store, uploaded_file
                            )

                        if result["status"] == "success":
                            st.markdown(
                                f"""
                            <div class="success-box">
                                <strong>Upload Successful!</strong><br>
                                File: {result["filename"]}<br>
                                Size: {result["size_bytes"] / 1024:.2f} KB<br>
                                File ID: {result["file_id"]}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.error(
                                f"Upload failed: {result.get('message', 'Unknown error')}"
                            )

                    except Exception as e:
                        st.error(f"Error uploading file: {str(e)}")

    else:  # Batch upload
        st.subheader("Upload Multiple Files")

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Maximum file size per file: {Config.MAX_FILE_SIZE_MB}MB",
        )

        if uploaded_files:
            # Display files info
            st.write(f"**Selected Files:** {len(uploaded_files)}")
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.write(f"**Total Size:** {total_size:.2f} MB")

            # List files
            with st.expander("View Files"):
                for f in uploaded_files:
                    st.write(f"• {f.name} ({f.size / 1024:.2f} KB)")

            if st.button("Upload All Files", type="primary"):
                try:
                    with st.spinner(f"Uploading {len(uploaded_files)} files..."):
                        # Reset file pointers
                        for f in uploaded_files:
                            f.seek(0)

                        result = api.batch_upload_files(
                            st.session_state.current_vector_store, uploaded_files
                        )

                    # Display results
                    st.markdown(
                        f"""
                    <div class="success-box">
                        <strong>Batch Upload Complete!</strong><br>
                        Total Files: {result["total_files"]}<br>
                        Successful: {result["successful_uploads"]}<br>
                        Failed: {result["failed_uploads"]}<br>
                        Processing Time: {result["processing_time_seconds"]:.2f}s
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Show individual results
                    if result["failed_uploads"] > 0:
                        with st.expander("View Failed Uploads"):
                            for res in result["results"]:
                                if res["status"] == "failed":
                                    st.error(
                                        f"• {res['filename']}: {res.get('message', 'Unknown error')}"
                                    )

                except Exception as e:
                    st.error(f"Error uploading files: {str(e)}")


# Page: Search
def page_search():
    """Vector store search page."""
    st.markdown(
        '<div class="main-header">Search Documents</div>', unsafe_allow_html=True
    )

    # Check if vector store is selected
    if not st.session_state.current_vector_store:
        st.warning("Please select a vector store first from the 'Vector Stores' page.")
        return

    # Display selected vector store
    try:
        store = api.get_vector_store(st.session_state.current_vector_store)
        st.markdown(
            f"""
        <div class="info-box">
            <strong>Searching in:</strong> {store["name"]}<br>
            <strong>Documents:</strong> {store["file_count"]}
        </div>
        """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return

    st.divider()

    # Search interface
    col1, col2 = st.columns([4, 1])

    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter your search query...",
            label_visibility="collapsed",
        )

    with col2:
        max_results = st.number_input(
            "Max Results", min_value=1, max_value=20, value=5, step=1
        )

    if st.button("Search", type="primary", use_container_width=True):
        if not search_query:
            st.warning("Please enter a search query")
        else:
            try:
                with st.spinner("Searching..."):
                    results = api.search(
                        search_query, st.session_state.current_vector_store, max_results
                    )

                st.success(
                    f"Found {results['total_results']} results in {results['processing_time_seconds']:.2f}s"
                )

                # Display results
                for i, result in enumerate(results["results"], 1):
                    with st.container():
                        st.markdown(f"### Result {i}")

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"** {result['filename']}**")

                        with col2:
                            st.metric("Relevance", f"{result['score']:.2%}")

                        st.markdown("**Content Preview:**")
                        st.text_area(
                            f"content_{i}",
                            result["content"],
                            height=150,
                            label_visibility="collapsed",
                        )

                        st.divider()

            except Exception as e:
                st.error(f"Search failed: {str(e)}")


# Page: Chat Interface
def page_chat():
    """Interactive chat interface."""
    st.markdown(
        '<div class="main-header"> Chat with Documents</div>', unsafe_allow_html=True
    )

    # Check if vector store is selected
    if not st.session_state.current_vector_store:
        st.warning("Please select a vector store first from the 'Vector Stores' page.")
        return

    # Display selected vector store
    try:
        store = api.get_vector_store(st.session_state.current_vector_store)

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown(f"**Vector Store:** {store['name']}")

        with col2:
            st.markdown(f"**Documents:** {store['file_count']}")

        with col3:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()

    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return

    st.divider()

    # Settings
    with st.expander("Settings"):
        col1, col2 = st.columns(2)

        with col1:
            use_streaming = st.checkbox("Stream Responses", value=False)

        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Higher values = more creative, Lower values = more focused",
            )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Response Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(
                            f"**Processing Time:** {message['metadata'].get('processing_time', 'N/A')}"
                        )
                    with col2:
                        st.write(
                            f"**Model:** {message['metadata'].get('model', 'N/A')}"
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            try:
                if use_streaming:
                    # Streaming response
                    response_placeholder = st.empty()
                    full_response = ""

                    response = api.stream_query(
                        prompt,
                        st.session_state.current_vector_store,
                        st.session_state.user_id,
                        st.session_state.session_id,
                    )

                    for chunk in response.iter_content(
                        chunk_size=None, decode_unicode=True
                    ):
                        if chunk:
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    # Add to messages
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "metadata": {
                                "processing_time": "N/A (streaming)",
                                "model": "gpt-4.1",
                            },
                        }
                    )

                else:
                    # Standard response
                    with st.spinner("Thinking..."):
                        response = api.query(
                            prompt,
                            st.session_state.current_vector_store,
                            st.session_state.user_id,
                            st.session_state.session_id,
                            temperature,
                        )

                    st.markdown(response["answer"])

                    # Store session ID for conversation history
                    if response.get("session_id"):
                        st.session_state.session_id = response["session_id"]

                    # Add to messages
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["answer"],
                            "metadata": {
                                "processing_time": f"{response['processing_time_seconds']:.2f}s",
                                "model": response.get("metadata", {}).get(
                                    "model", "N/A"
                                ),
                                "turn_count": response.get("turn_count", "N/A"),
                            },
                        }
                    )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "metadata": {}}
                )


# Main App
def main():
    """Main application."""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80
        )
        st.title("Knowledge Assistant")
        st.caption("RAG-powered document Q&A")

        st.divider()

        # Health status
        display_health_status()

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            ["Chat", "Search", "Upload", "Vector Stores"],
            label_visibility="collapsed",
        )

        st.divider()

        # Current selection
        if st.session_state.current_vector_store:
            st.success("Vector Store Selected")
            try:
                store = api.get_vector_store(st.session_state.current_vector_store)
                st.caption(f"{store['name']}")
                st.caption(f"{store['file_count']} files")
            except:
                pass
        else:
            st.info("No vector store selected")

        st.divider()

        # Info
        st.caption("SAI Cookbook")

    # Main content
    if page == "Chat":
        page_chat()
    elif page == "Search":
        page_search()
    elif page == "Upload":
        page_file_upload()
    elif page == "Vector Stores":
        page_vector_stores()


if __name__ == "__main__":
    main()
