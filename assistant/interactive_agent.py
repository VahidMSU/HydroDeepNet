from llama_index.core import VectorStoreIndex, Settings
try:
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
except ImportError:
    print("Ollama is not installed")
from llama_index.core.schema import TextNode
from pathlib import Path
import json
import os
import concurrent.futures
import multiprocessing
import uuid
try:
    from .Logger import LoggerSetup
    from .memory_system import MemorySystem
    from .query_understanding import QueryUnderstanding
    from .response_generator import ResponseGenerator
    from .UserQueryAgent import UserQueryAgent
    from .utils import describe_markdown, describe_image, summarize_csv
    from .discover_reports import discover_reports
except ImportError:
    # Fallback for direct script execution
    from assistant.Logger import LoggerSetup
    from assistant.memory_system import MemorySystem
    from assistant.query_understanding import QueryUnderstanding
    from assistant.response_generator import ResponseGenerator
    from assistant.UserQueryAgent import UserQueryAgent
    from assistant.utils import describe_markdown, describe_image, summarize_csv
    from assistant.discover_reports import discover_reports


# ==== FILE INDEXING WITH PARALLEL PROCESSING ====
def load_all_documents(report_dir, logger, only_md_files=True, max_workers=4):

    logger.info(f"Loading documents with parallel processing...")

    # If only loading markdown files, use a file filter
    if only_md_files:
        # Create a custom file filter for SimpleDirectoryReader
        def md_file_filter(file_path):
            return file_path.endswith('.md')

        logger.info("Loading only markdown files for faster indexing")
        file_filter = md_file_filter
    else:
        logger.info("Loading all document types")
        file_filter = None

    # Get all files that match our filter
    all_files = []
    for root, _, files in os.walk(report_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_filter is None or file_filter(file_path):
                all_files.append(file_path)

    logger.info(f"Found {len(all_files)} files matching filter criteria")
    if not all_files:
        return []

    # Function to load a single file
    def load_single_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        file_name = os.path.basename(file_path)
        # Create a TextNode with the file content
        return TextNode(text=content, metadata={"file_name": file_name, "file_path": file_path})

    # Use parallel processing to load files
    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for doc in executor.map(load_single_file, all_files):
            if doc is not None:
                documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents using parallel processing")
    return documents


def index_documents(documents, logger, max_workers=4):
    if not documents:
        logger.warning("No documents to index")
        return None

    logger.info(f"Indexing {len(documents)} documents with parallel processing...")

    # Print document structure for debugging
    logger.info(f"Document sample: {str(documents[0].__dict__)[:200]}..." if documents else "No documents")

    # Configure Settings instead of using deprecated ServiceContext
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20

    # Check if embeddings model is configured correctly
    assert hasattr(Settings, 'embed_model') and Settings.embed_model is not None, "Embeddings model is not configured. Check Settings.embed_model"

    logger.info(f"Using embedding model: {type(Settings.embed_model).__name__}")

    # Import the Node base class


    # Create properly configured TextNodes from our documents
    processed_nodes = []

    for doc in documents:
        # Get the content
        content = doc.get_content() if hasattr(doc, 'get_content') else str(doc)

        # Get the document ID or create one
        doc_id = getattr(doc, 'id_', None) or str(uuid.uuid4())

        # Get the metadata or create empty dict
        metadata = getattr(doc, 'metadata', {}).copy()

        # Add hash to metadata
        metadata['hash'] = str(hash(content + str(metadata)))
        metadata['node_id'] = doc_id

        # Create a new TextNode with all the required attributes
        node = TextNode(
            text=content,
            id_=doc_id,
            metadata=metadata
        )

        processed_nodes.append(node)

    # Log the processed nodes for debugging
    logger.info(f"Created {len(processed_nodes)} TextNode objects")

    # Create the index using the constructor approach
    logger.info("Creating VectorStoreIndex with TextNode objects")
    index = VectorStoreIndex(processed_nodes, show_progress=True)

    logger.info(f"Successfully indexed {len(documents)} documents")
    return index

# ==== PRELOAD MEMORY WITH IMPORTANT FILES ====
def preload_memory(memory, report_structure, logger, DEFAULT_MODEL):
    """
    Preload important files into memory to enhance the system's knowledge
    before user interaction begins.

    Args:
        memory: MemorySystem instance
        report_structure: Dictionary containing report file structure
        logger: Logger instance
        DEFAULT_MODEL: Model to use for analysis
    """
    logger.info("Preloading memory with important files...")

    # Track what's been preloaded
    preloaded_files = {
        'csv': [],
        'image': [],
        'markdown': [],
        'html': []
    }

    # Skip CSV files preloading
    logger.info("Skipping CSV files preloading...")

    # Skip image files preloading
    logger.info("Skipping image files preloading...")

    # 3. Preload markdown files (reports, documentation)
    logger.info("Preloading markdown files...")
    md_files = []

    # Find all markdown files in the report structure
    for group_name, group_data in report_structure.items():
        # With the simplified structure, files are in the 'files' dictionary
        for file_name, file_data in group_data.get('files', {}).items():
            if file_name.lower().endswith('.md'):
                file_path = file_data.get('path')
                if file_path:
                    md_files.append((file_name, file_path, group_name))

    # Process markdown files
    for file_name, file_path, group_name in md_files:
        logger.info(f"Preloading markdown: {file_name} from {group_name}")
        result = describe_markdown(file_path, DEFAULT_MODEL, logger, memory)
        preloaded_files['markdown'].append(file_name)

        # Add a general note about this document to memory
        query_info = {
            "keywords": ["document", "markdown", file_name, group_name],
            "intent": "search"
        }
        memory.store_interaction(
            f"What information is in the {file_name} document?",
            f"The document {file_name} in {group_name} contains: {result[:200]}...",
            query_info,
            [file_path]
        )

    # Log summary of what was preloaded
    preload_summary = []
    for file_type, files in preloaded_files.items():
        if files:
            preload_summary.append(f"Preloaded {len(files)} {file_type} files")

    logger.info("Memory preloading complete: " + ", ".join(preload_summary))

    # Return summary of what was preloaded
    return preloaded_files

def read_report_structure(REPORT_DIR):
    with open(Path(REPORT_DIR) / "report_structure.json") as f:
        report_structure = json.load(f)
    return report_structure

def run_chat(logger, REPORT_DIR, DEFAULT_MODEL, report_structure=None, max_workers=100):
    # Initialize the memory system
    base_path = "/data/SWATGenXApp/codes/assistant"
    logger.info("Indexing documents, please wait...")

    query_engine = index_documents(documents=load_all_documents(REPORT_DIR, logger, only_md_files=True, max_workers=max_workers), logger=logger, max_workers=max_workers).as_query_engine()

    # Initialize the user query agent
    user_agent = UserQueryAgent(
        memory=MemorySystem(f"{base_path}/memory", embedding_model_name="nomic-embed-text"),
        query_understanding=QueryUnderstanding(default_model=DEFAULT_MODEL),
        response_generator=ResponseGenerator(llm_service=None),
        report_structure=read_report_structure(REPORT_DIR),
        logger=logger,
        DEFAULT_MODEL=DEFAULT_MODEL
    )

    # Return the user agent and query engine
    return user_agent, query_engine


def interactive_agent(username="admin"):
    from datetime import datetime
    import re
    report_id = os.listdir(f"/data/SWATGenXApp/Users/{username}/Reports/")

    # Filter out files that don't match the date format pattern before sorting
    date_pattern = re.compile(r'^\d{8}_\d{6}$')
    report_id = [rid for rid in report_id if date_pattern.match(rid)]

    # Sort and get the latest report
    if report_id:
        report_id = sorted(report_id, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S"))[-1]
        logger = LoggerSetup(rewrite=True, verbose=True)
        # ==== CONFIGURATION ====
        REPORT_DIR = f"/data/SWATGenXApp/Users/{username}/Reports/{report_id}"
        DEFAULT_MODEL = "Llama3.2-Vision:latest"
        DEFAULT_EMBED_MODEL = "nomic-embed-text"

        # Determine optimal number of workers for parallel processing
        num_cores = multiprocessing.cpu_count()
        max_workers = min(num_cores, 8)  # Use at most 8 workers to avoid overloading
        logger.info(f"Using {max_workers} workers for parallel processing (system has {num_cores} cores)")

        # ==== SETUP ====
        Settings.embed_model = OllamaEmbedding(model_name=DEFAULT_EMBED_MODEL)
        Settings.llm = Ollama(model=DEFAULT_MODEL)
        base_path = "/data/SWATGenXApp/codes/assistant"
        os.makedirs(f"{base_path}/memory", exist_ok=True)

        # ==== Run the chat interface ====
        report_structure = discover_reports(base_dir=REPORT_DIR)

        # Use the pre-loaded report structure instead of reading from disk again
        return run_chat(logger, REPORT_DIR, DEFAULT_MODEL,
                report_structure=report_structure, max_workers=max_workers)
    else:
        return None, None

    # ==== END OF SCRIPT ====

if __name__ == "__main__":
    username = "admin"
    user_agent, query_engine = interactive_agent(username)

    if user_agent and query_engine:
        # Interactive chat loop at the higher level
        while True:
            user_input = input("\n>> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break

            # Process query through the UserQueryAgent
            response = user_agent.process_query(user_input, query_engine)

            # If the response is None, it means the command output was already printed
            # or the command wasn't handled and should be processed by the default logic
            if response is not None:
                print(f"Response: {response}")
    else:
        print("No valid report directories found. Please check the reports directory.")
