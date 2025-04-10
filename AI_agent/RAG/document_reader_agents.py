import os
import re
import json
import logging
import traceback
from agno.agent import Agent

from agno.knowledge.combined import CombinedKnowledgeBase
from agno.knowledge.langchain import LangChainKnowledgeBase
from typing import Dict, Any, List, Optional
from Logger import LoggerSetup
# Initialize logger using setup_logger method
logger_setup = LoggerSetup()
logger = logger_setup.setup_logger()

def get_agno_agent(system_message=None, model="gpt-4o", temperature=0, **kwargs):
    """Create and return an Agno agent with the specified parameters.
    
    Args:
        system_message (str, optional): The system message or instructions for the agent.
        model (str, optional): The model to use. Defaults to "gpt-4o".
        temperature (float, optional): Temperature setting for the model. Defaults to 0.
        **kwargs: Additional arguments to pass to the Agent constructor.
        
    Returns:
        Agent: An initialized Agno agent.
    """
    try:
        if system_message:
            # Create agent with system message (instructions)
            agent = Agent(
                instructions=system_message,
                model=model,
                temperature=temperature,
                **kwargs
            )
        else:
            # Create agent without system message
            agent = Agent(
                model=model,
                temperature=temperature,
                **kwargs
            )
        
        logger.info(f"Created Agno agent with model: {model}")
        return agent
        
    except Exception as e:
        logger.error(f"Error creating Agno agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise

class AgentManager:
    """Manages the multi-agent system for the document reader."""
    
    def __init__(self, document_reader):
        """Initialize the agent manager with reference to the document reader."""
        self.document_reader = document_reader
        self.model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.interactive_agent = None
        self.visual_analyst_agent = None
        self.document_navigator_agent = None
        self.data_scientist_agent = None
        self.knowledge_graph_builder = None
    
    def initialize_interactive_agent(self) -> None:
        """Initialize the main interactive agent."""
        try:
            logger.info("Initializing interactive agent...")
            
            # Define the system prompt
            system_prompt = """You are a helpful assistant that helps users understand and interpret data sources and results.
            You can help users analyze CSV data, interpret images, and summarize content from markdown and text files.
            
            You have access to multiple file types:
            - CSV files: You can analyze tabular data, run basic statistics, and extract insights.
            - Markdown (.md) files: You can read and summarize text content, as well as extract code examples.
            - Image files: You can describe and interpret images, charts, and diagrams.
            - Text (.txt) files: You can process and summarize plain text content.
            
            When asked about data or files, always check what files are available before responding.
            If you're unsure about something, ask clarifying questions.
            
            Available commands:
            - To view a CSV file: "show me data from [filename]" or "analyze [filename]"
            - To visualize data: "visualize [column] from [filename]" or "plot [column] vs [column] from [filename]"
            - To read a markdown file: "read [filename]" or "summarize [filename]"
            - To analyze an image: "analyze image [filename]" or "describe [filename]"
            - To get a summary of available data: "what data is available?" or "list files"
            
            Always provide helpful, accurate information, and guide the user through their data analysis journey.
            """
            
            # Initialize the agent using Agno
            self.interactive_agent = get_agno_agent(
                system_message=system_prompt,
                model=self.model_name,
                temperature=0
            )
            
            # Store the system prompt
            self.document_reader.context['system_prompt'] = system_prompt
            
            logger.info("Interactive agent initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing interactive agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def initialize_multi_agent_system(self) -> None:
        """Initialize the multi-agent system with specialized agents."""
        try:
            logger.info("Initializing multi-agent system...")
            
            # Initialize the visual analyst agent
            self._initialize_visual_analyst_agent()
            
            # Initialize the document navigator agent
            self._initialize_document_navigator_agent()
            
            # Initialize the data scientist agent
            self._initialize_data_scientist_agent()
            
            # Initialize the knowledge graph builder (if needed)
            # self._initialize_knowledge_graph_builder()
            
            logger.info("Multi-agent system initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing multi-agent system: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _initialize_visual_analyst_agent(self) -> None:
        """Initialize the agent specialized in visual analysis."""
        try:
            logger.info("Initializing visual analyst agent...")
            
            # Define the system prompt for visual analysis
            visual_system_prompt = """You are a visual analyst agent specialized in analyzing images, charts, and visualizations.
            Your primary responsibility is to interpret images and extract insights from visual content.
            
            When analyzing an image:
            1. Describe what the image shows in detail
            2. Identify any text content in the image
            3. For charts and graphs:
               - Identify the type of chart/graph
               - Describe the axes and their units
               - Identify trends, patterns, and key data points
               - Summarize the main insights from the visualization
            4. For technical diagrams:
               - Identify components and their relationships
               - Explain the system or process depicted
               - Highlight key elements and their functions
            5. For photographs:
               - Describe the subject matter in detail
               - Note relevant environmental or contextual elements
               - Identify any text or labels
            
            Always be precise, technical, and detailed in your analysis. Focus on extracting actionable information
            and insights from the image rather than just describing what it looks like.
            """
            
            # Initialize the agent using Agno
            self.visual_analyst_agent = get_agno_agent(
                system_message=visual_system_prompt,
                model=self.model_name,
                temperature=0
            )
            
            logger.info("Visual analyst agent initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing visual analyst agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _initialize_document_navigator_agent(self) -> None:
        """Initialize the agent specialized in navigating and extracting information from documents."""
        try:
            logger.info("Initializing document navigator agent...")
            
            # Define the system prompt for document navigation
            document_navigator_prompt = """You are a document navigator agent specialized in extracting and organizing information from various document types.
            Your primary responsibility is to help users find specific information within documents and summarize document content.
            
            Your capabilities include:
            1. Summarizing the content of markdown and text files
            2. Extracting key sections from documents
            3. Finding specific information based on user queries
            4. Identifying main topics and themes in documents
            5. Organizing information into structured formats
            
            When analyzing documents:
            - Focus on extracting the most relevant information based on the user's query
            - Provide concise summaries that capture the main points
            - Highlight key findings, conclusions, or recommendations
            - Identify connections between different parts of the document
            - Maintain the context and original meaning of the information
            
            Always be accurate and comprehensive in your analysis, while presenting information in a clear,
            structured format that is easy for the user to understand.
            """
            
            # Initialize the agent using Agno
            self.document_navigator_agent = get_agno_agent(
                system_message=document_navigator_prompt,
                model=self.model_name,
                temperature=0
            )
            
            logger.info("Document navigator agent initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing document navigator agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _initialize_data_scientist_agent(self) -> None:
        """Initialize the agent specialized in data analysis and visualization."""
        try:
            logger.info("Initializing data scientist agent...")
            
            # Define the system prompt for data analysis
            data_scientist_prompt = """You are a data scientist agent specialized in analyzing and visualizing tabular data.
            Your primary responsibility is to help users extract insights from CSV files and create meaningful visualizations.
            
            Your capabilities include:
            1. Analyzing tabular data for trends, patterns, and outliers
            2. Calculating basic statistics (mean, median, mode, std dev, etc.)
            3. Suggesting appropriate visualizations based on data types
            4. Interpreting the results of data analysis
            5. Providing insights and recommendations based on data
            
            When analyzing data:
            - Focus on understanding the data structure and types
            - Identify key variables and their relationships
            - Look for trends, patterns, anomalies, and outliers
            - Consider the context and domain of the data
            - Provide actionable insights based on the analysis
            
            Always be thorough and methodical in your analysis, while explaining your findings
            in a clear, concise manner that non-technical users can understand.
            """
            
            # Initialize the agent using Agno
            self.data_scientist_agent = get_agno_agent(
                system_message=data_scientist_prompt,
                model=self.model_name,
                temperature=0
            )
            
            logger.info("Data scientist agent initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing data scientist agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _initialize_knowledge_graph_builder(self) -> None:
        """Initialize the knowledge graph builder agent (if needed)."""
        try:
            logger.info("Initializing knowledge graph builder agent...")
            
            # Define the system prompt for knowledge graph building
            knowledge_graph_prompt = """You are a knowledge graph builder agent specialized in extracting entities and relationships from text.
            Your primary responsibility is to identify key concepts, entities, and the relationships between them.
            
            Your capabilities include:
            1. Identifying entities (people, organizations, locations, concepts, etc.)
            2. Determining relationships between entities
            3. Building a structured knowledge representation
            4. Answering questions based on the knowledge graph
            5. Suggesting connections and insights based on the graph structure
            
            When building a knowledge graph:
            - Focus on extracting meaningful entities and relationships
            - Organize information in a structured, interconnected format
            - Prioritize accuracy and relevance of connections
            - Consider the context and domain-specific meaning
            - Build connections across different documents when possible
            
            Always be precise and thorough in your extraction, while maintaining the semantic meaning
            of the original content and presenting information in an accessible way.
            """
            
            # Initialize the agent using Agno
            self.knowledge_graph_builder = get_agno_agent(
                system_message=knowledge_graph_prompt,
                model=self.model_name,
                temperature=0
            )
            
            logger.info("Knowledge graph builder agent initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge graph builder agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_chat_response(self, message: str) -> str:
        """Get response from the interactive agent based on the message and context."""
        try:
            # Check if the interactive agent is initialized
            if not self.interactive_agent:
                return "Error: Interactive agent not initialized."
            
            # Build the history from the conversation context
            history = self.document_reader.context.get('conversation_history', [])
            
            # Run the agent with the message and history
            response = self.interactive_agent.chat(
                message,
                history=[
                    (entry['content'], entry['role']) 
                    for entry in history
                ]
            )
            
            # Return the response content
            return response
            
        except Exception as e:
            logger.error(f"Error getting chat response: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}"
    
    def run_agent(self, agent_type: str, prompt: str, **kwargs) -> str:
        """Run a specific agent with the given prompt and additional arguments."""
        try:
            # Determine which agent to use
            agent = None
            if agent_type == 'visual_analyst':
                agent = self.visual_analyst_agent
            elif agent_type == 'document_navigator':
                agent = self.document_navigator_agent
            elif agent_type == 'data_scientist':
                agent = self.data_scientist_agent
            elif agent_type == 'knowledge_graph_builder':
                agent = self.knowledge_graph_builder
            elif agent_type == 'interactive':
                return self.get_chat_response(prompt)
            else:
                return f"Error: Unknown agent type '{agent_type}'"
            
            # Check if the agent is initialized
            if not agent:
                return f"Error: Agent '{agent_type}' not initialized."
            
            # Run the agent with the prompt and additional arguments
            response = agent.run(prompt, **kwargs)
            
            # Clean the response if needed
            if hasattr(self.document_reader, 'response_handler'):
                response = self.document_reader.response_handler.clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error running agent {agent_type}: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error running agent: {str(e)}"

class AgentHandler:
    """Handles agent initialization and management for the InteractiveDocumentReader."""
    
    def __init__(self, document_reader):
        """Initialize the agent handler with a reference to the document_reader."""
        self.document_reader = document_reader
    
    def initialize_multi_agent_system(self):
        """Initialize the multi-agent system with specialized agents."""
        try:
            # Define the multi-agent system
            self.document_reader.agents = {
                "coordinator": None,
                "visual_analyst": None,
                "document_navigator": None,
                "data_scientist": None
            }
            
            # Initialize the coordinator agent
            self.document_reader.agents["coordinator"] = Agent(
                instructions="""
                You are the coordinator of a multi-agent AI system designed to help users understand and analyze documents.
                Your role is to:
                1. Determine which specialist agent should handle specific user requests
                2. Synthesize responses from multiple agents when needed
                3. Provide helpful and coherent answers to general questions
                
                Be concise, direct, and focused on providing valuable information.
                """,
                model="gpt-4o"
            )
            
            # Initialize visual analyst agent
            self.document_reader.agents["visual_analyst"] = Agent(
                instructions="""
                You are a visual analyst specialized in interpreting images, charts, and visualizations.
                Your expertise includes:
                - Analyzing charts and graphs to extract insights
                - Interpreting scientific visualizations and plots
                - Identifying trends, patterns, and anomalies in visual data
                - Explaining complex visualizations in clear, accessible language
                
                For charts and graphs, always include:
                1. What type of visualization it is
                2. What data is being shown
                3. The key patterns or insights visible
                4. Any limitations or potential misinterpretations
                
                Focus on providing analytical insights rather than just descriptions.
                """,
                model="gpt-4o-vision"
            )
            
            # Initialize document navigator agent
            self.document_reader.agents["document_navigator"] = Agent(
                instructions="""
                You are a document navigator specialized in retrieving, organizing, and summarizing text-based information.
                Your expertise includes:
                - Finding relevant documents and passages based on user queries
                - Summarizing key information from long documents
                - Extracting structured information from unstructured text
                - Helping users understand complex documentation
                
                For any document-related queries, provide:
                1. Clear, concise summaries
                2. Relevant excerpts when helpful
                3. Context for understanding the information
                
                Focus on accuracy and clarity in your responses.
                """,
                model="gpt-4o"
            )
            
            # Initialize data scientist agent
            self.document_reader.agents["data_scientist"] = Agent(
                instructions="""
                You are a data scientist specialized in analyzing numerical data and statistics.
                Your expertise includes:
                - Interpreting statistical analyses and research findings
                - Explaining data patterns and correlations
                - Evaluating the significance and implications of data points
                - Suggesting possible approaches for deeper data analysis
                
                For data-related queries, provide:
                1. Clear interpretations of what the data shows
                2. Potential causal relationships (with appropriate caution)
                3. Limitations or caveats about the data
                4. Suggestions for further analysis when appropriate
                
                Focus on making complex data understandable while maintaining scientific rigor.
                """,
                model="gpt-4o"
            )
            
            logger.info("Multi-agent system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing multi-agent system: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def initialize_interactive_agent(self):
        """Initialize the main interactive agent with comprehensive instructions."""
        try:
            # Define base instructions
            instructions = """
            You are an AI assistant specializing in helping users understand and interpret various documents and data sources.
            
            Your capabilities include:
            - Analyzing CSV data for patterns and insights
            - Interpreting images, charts, and visualizations
            - Summarizing content from markdown and text files
            - Analyzing JSON data structures
            
            When responding to users:
            - Be concise and direct
            - Provide specific insights rather than general statements
            - Use formatting to organize information clearly
            - Acknowledge limitations if you cannot fully answer a question
            
            You have access to these data sources:
            """
            
            # Add information about discovered files
            if self.document_reader.discovered_files:
                for file_type, files in self.document_reader.discovered_files.items():
                    if files:
                        instructions += f"- {len(files)} {file_type.upper()} files\n"
            else:
                instructions += "- No files have been discovered yet. Ask the user to use /discover to find files.\n"
            
            # Initialize the interactive agent
            self.document_reader.interactive_agent = Agent(
                instructions=instructions,
                model="gpt-4o"
            )
            
            logger.info("Interactive agent initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing interactive agent: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def initialize_image_reader(self, doc_type, config):
        """Initialize an image reader agent for analyzing images."""
        try:
            # Extract image paths from config
            image_paths = []
            for item in config:
                if 'path' in item and os.path.exists(item['path']):
                    image_paths.append(item['path'])
            
            if not image_paths:
                logger.warning("No valid image paths provided for image reader")
                return False
            
            # Initialize the image agent
            self.document_reader.image_agent = Agent(
                instructions="""
                You are an AI assistant specializing in analyzing and interpreting images.
                
                Your capabilities include:
                - Describing the content of images in detail
                - Analyzing charts, graphs, and visualizations
                - Identifying key elements and patterns in images
                - Explaining technical diagrams and scientific visualizations
                
                When analyzing images:
                - Provide a comprehensive description of what you see
                - For charts and graphs, identify the type of visualization and what data is being shown
                - Extract any visible metrics, values, or trends
                - Explain the significance of the visualization when possible
                
                Be precise, thorough, and focus on the most important aspects of each image.
                """,
                model="gpt-4o-vision"
            )
            
            # Store image paths
            for path in image_paths:
                from agno.media import Image
                self.document_reader.images.append(Image(filepath=path))
            
            logger.info(f"Image reader initialized with {len(image_paths)} images")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing image reader: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def initialize_knowledge_base(self, doc_type, config):
        """Initialize a knowledge base for a specific document type."""
        try:
            # Extract document paths from config
            doc_paths = []
            for item in config:
                if 'path' in item and os.path.exists(item['path']):
                    doc_paths.append(item['path'])
            
            if not doc_paths:
                logger.warning(f"No valid {doc_type} paths provided for knowledge base")
                return False
            
            # Initialize the specific knowledge base type
            if doc_type == 'csv':
                from agno.knowledge import CSVKnowledgeBase
                self.document_reader.knowledge_bases[doc_type] = CSVKnowledgeBase(
                    filepaths=doc_paths
                )
            elif doc_type == 'pdf':
                from agno.knowledge import PDFKnowledgeBase
                self.document_reader.knowledge_bases[doc_type] = PDFKnowledgeBase(
                    filepaths=doc_paths
                )
            elif doc_type == 'text':
                from agno.knowledge import TextKnowledgeBase
                self.document_reader.knowledge_bases[doc_type] = TextKnowledgeBase(
                    filepaths=doc_paths
                )
            elif doc_type == 'json':
                from agno.knowledge import JSONKnowledgeBase
                self.document_reader.knowledge_bases[doc_type] = JSONKnowledgeBase(
                    filepaths=doc_paths
                )
            elif doc_type == 'docx':
                from agno.knowledge import DocxKnowledgeBase
                self.document_reader.knowledge_bases[doc_type] = DocxKnowledgeBase(
                    filepaths=doc_paths
                )
            elif doc_type == 'website':
                from agno.knowledge import WebsiteKnowledgeBase
                urls = [item.get('url') for item in config if 'url' in item]
                if urls:
                    self.document_reader.knowledge_bases[doc_type] = WebsiteKnowledgeBase(
                        urls=urls
                    )
            # Add LangChain integration example
            elif doc_type == 'langchain_retriever':
                # Example of how to use a LangChain retriever as a knowledge base
                retriever_config = next((item for item in config if 'retriever' in item), None)
                if retriever_config and 'retriever' in retriever_config:
                    retriever = retriever_config['retriever']
                    self.document_reader.knowledge_bases[doc_type] = LangChainKnowledgeBase(
                        retriever=retriever
                    )
            else:
                logger.warning(f"Unsupported document type: {doc_type}")
                return False
            
            logger.info(f"{doc_type.upper()} knowledge base initialized with {len(doc_paths)} files")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {doc_type} knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def create_combined_knowledge_base(self):
        """Create a combined knowledge base from all initialized knowledge bases."""
        try:
            if len(self.document_reader.knowledge_bases) < 2:
                logger.warning("Not enough knowledge bases to create a combined knowledge base")
                return False
            
            # Create a combined knowledge base
            combined_knowledge = CombinedKnowledgeBase(
                knowledge_bases=list(self.document_reader.knowledge_bases.values())
            )
            
            # Create an agent with the combined knowledge base
            self.document_reader.agent = Agent(
                knowledge=combined_knowledge,
                search_knowledge=True
            )
            
            logger.info(f"Combined knowledge base created with {len(self.document_reader.knowledge_bases)} sources")
            return True
            
        except Exception as e:
            logger.error(f"Error creating combined knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return False 