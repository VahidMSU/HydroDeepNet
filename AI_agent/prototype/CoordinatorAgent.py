from typing import Dict, List, Any, Optional, Tuple
import re
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import logging

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from Logger import Logger

class QueryType(Enum):
    VISUAL = "visual"
    DATA = "data"
    DOCUMENT = "document"
    DOMAIN = "domain"
    GENERAL = "general"

@dataclass
class AgentResponse:
    content: str
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime

class CoordinatorAgent:
    """Coordinates the multi-agent system and manages query routing."""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 log_dir: str = "logs", 
                 log_level: int = logging.INFO):
        """
        Initialize the coordinator agent.
        
        Args:
            openai_api_key: Optional OpenAI API key
            log_dir: Directory for log files
        """
        # Set up logging
        self.logger = Logger(
            log_dir=log_dir,
            app_name="CoordinatorAgent"
        )
    
        self.logger.info("Initializing CoordinatorAgent")
        
        # Initialize specialized agents
        self.agents = self._initialize_agents()
        self.logger.info(f"Initialized {len(self.agents)} specialized agents")
        
        # Track agent performance
        self.agent_metrics = {
            agent_type: {
                "queries_handled": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0
            }
            for agent_type in self.agents.keys()
        }
        
        # Conversation context
        self.conversation_context = []
        self.max_context_length = 10
        
        self.logger.info("CoordinatorAgent initialization completed")

    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize the specialized agents."""
        self.logger.info("Initializing specialized agents")
        agents = {
            "visual_analyst": Agent(
                model=OpenAIChat(id="gpt-4o"),
                agent_id="visual-analyst",
                name="Visual Analyst",
                instructions=[
                    "You are a visual analysis expert.",
                    "Analyze images, charts, and visualizations.",
                    "Extract insights from visual data.",
                    "Explain patterns and trends."
                ]
            ),
            "data_scientist": Agent(
                model=OpenAIChat(id="gpt-4"),
                agent_id="data-scientist",
                name="Data Scientist",
                instructions=[
                    "You are a data analysis expert.",
                    "Analyze numerical data and statistics.",
                    "Identify patterns and correlations.",
                    "Provide statistical insights."
                ]
            ),
            "document_navigator": Agent(
                model=OpenAIChat(id="gpt-4"),
                agent_id="document-navigator",
                name="Document Navigator",
                instructions=[
                    "You are a document analysis expert.",
                    "Navigate and summarize documents.",
                    "Extract key information.",
                    "Provide relevant context."
                ]
            ),
            "domain_expert": Agent(
                model=OpenAIChat(id="gpt-4"),
                agent_id="domain-expert",
                name="Domain Expert",
                instructions=[
                    "You are a domain expert in hydrology and environmental science.",
                    "Provide domain-specific knowledge.",
                    "Explain technical concepts.",
                    "Connect data to domain context."
                ]
            )
        }
        self.logger.debug(f"Successfully initialized agents: {', '.join(agents.keys())}")
        return agents

    def route_query(self, 
                   query: str, 
                   context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Route a query to the appropriate agent(s).
        
        Args:
            query: The user's query
            context: Optional additional context
            
        Returns:
            AgentResponse with the final response
        """
        self.logger.info(f"Routing query: {query}")
        
        # Analyze query type
        query_type, confidence = self._analyze_query_type(query)
        self.logger.debug(f"Query type: {query_type.value}, confidence: {confidence}")
        
        # Update context
        self._update_context(query, query_type)
        
        # Get relevant agents
        primary_agent, supporting_agents = self._get_relevant_agents(query_type, query)
        self.logger.debug(f"Selected agents - Primary: {primary_agent}, Supporting: {supporting_agents}")
        
        # Get responses from agents
        responses = self._gather_agent_responses(
            query, 
            primary_agent, 
            supporting_agents,
            context
        )
        self.logger.debug(f"Received {len(responses)} agent responses")
        
        # Synthesize final response
        final_response = self._synthesize_responses(responses, query_type)
        self.logger.info(f"Generated response with confidence: {final_response.confidence}")
        
        # Update metrics
        self._update_metrics(query_type, responses)
        
        return final_response

    def _analyze_query_type(self, query: str) -> Tuple[QueryType, float]:
        """Determine the type of query and confidence level."""
        self.logger.debug(f"Analyzing query type: {query}")
        query_lower = query.lower()
        
        # Visual analysis patterns
        visual_patterns = [
            r'image', r'picture', r'graph', r'chart', r'plot', r'visualization',
            r'show', r'look', r'see', r'display'
        ]
        
        # Data analysis patterns
        data_patterns = [
            r'analyze', r'calculate', r'statistics', r'correlation', r'trend',
            r'numbers', r'data', r'average', r'mean', r'median'
        ]
        
        # Document patterns
        document_patterns = [
            r'document', r'text', r'read', r'summary', r'content',
            r'file', r'report', r'paper'
        ]
        
        # Domain-specific patterns
        domain_patterns = [
            r'hydrology', r'water', r'climate', r'environment', r'groundwater',
            r'aquifer', r'precipitation', r'temperature'
        ]
        
        # Calculate match scores
        scores = {
            QueryType.VISUAL: sum(bool(re.search(p, query_lower)) for p in visual_patterns),
            QueryType.DATA: sum(bool(re.search(p, query_lower)) for p in data_patterns),
            QueryType.DOCUMENT: sum(bool(re.search(p, query_lower)) for p in document_patterns),
            QueryType.DOMAIN: sum(bool(re.search(p, query_lower)) for p in domain_patterns)
        }
        
        # Get the type with highest score
        max_score = max(scores.values())
        if max_score == 0:
            self.logger.debug("No specific patterns matched, defaulting to GENERAL type")
            return QueryType.GENERAL, 0.5
            
        query_type = max(scores.items(), key=lambda x: x[1])[0]
        confidence = max_score / len(locals()[f"{query_type.value}_patterns"])
        
        self.logger.debug(f"Determined query type: {query_type.value} with confidence: {confidence}")
        return query_type, confidence

    def _get_relevant_agents(self, 
                           query_type: QueryType, 
                           query: str) -> Tuple[str, List[str]]:
        """Determine primary and supporting agents for the query."""
        self.logger.debug(f"Getting relevant agents for query type: {query_type.value}")
        
        # Default mappings
        primary_mappings = {
            QueryType.VISUAL: "visual_analyst",
            QueryType.DATA: "data_scientist",
            QueryType.DOCUMENT: "document_navigator",
            QueryType.DOMAIN: "domain_expert"
        }
        
        # Get primary agent
        primary_agent = primary_mappings.get(query_type, "document_navigator")
        
        # Determine supporting agents based on query content
        supporting_agents = []
        
        # Add domain expert for technical queries
        if any(term in query.lower() for term in [
            "why", "how", "explain", "meaning", "significance"
        ]):
            supporting_agents.append("domain_expert")
        
        # Add data scientist for numerical queries
        if any(term in query.lower() for term in [
            "calculate", "compare", "trend", "statistics"
        ]):
            supporting_agents.append("data_scientist")
        
        self.logger.debug(f"Selected primary agent: {primary_agent}")
        self.logger.debug(f"Selected supporting agents: {supporting_agents}")
        return primary_agent, supporting_agents

    def _gather_agent_responses(self,
                              query: str,
                              primary_agent: str,
                              supporting_agents: List[str],
                              context: Optional[Dict[str, Any]] = None) -> List[AgentResponse]:
        """Gather responses from relevant agents."""
        self.logger.info("Gathering agent responses")
        responses = []
        
        # Get primary agent response
        self.logger.debug(f"Getting response from primary agent: {primary_agent}")
        primary_response = self.agents[primary_agent].print_response(
            query,
            context=context,
            stream=False
        )
        
        responses.append(AgentResponse(
            content=primary_response,
            confidence=0.9,  # Primary agent gets high confidence
            metadata={"agent": primary_agent, "role": "primary"},
            timestamp=datetime.now()
        ))
        
        # Get supporting agent responses
        for agent_name in supporting_agents:
            if agent_name in self.agents and agent_name != primary_agent:
                self.logger.debug(f"Getting response from supporting agent: {agent_name}")
                support_response = self.agents[agent_name].print_response(
                    f"Provide supporting information for: {query}",
                    context=context,
                    stream=False
                )
                
                responses.append(AgentResponse(
                    content=support_response,
                    confidence=0.7,  # Supporting agents get lower confidence
                    metadata={"agent": agent_name, "role": "support"},
                    timestamp=datetime.now()
                ))
        
        self.logger.debug(f"Gathered {len(responses)} responses")
        return responses

    def _synthesize_responses(self, 
                            responses: List[AgentResponse],
                            query_type: QueryType) -> AgentResponse:
        """Synthesize multiple agent responses into a coherent response."""
        self.logger.info("Synthesizing agent responses")
        
        if not responses:
            self.logger.warning("No agent responses to synthesize")
            return AgentResponse(
                content="I apologize, but I couldn't generate a response.",
                confidence=0.0,
                metadata={"error": "No agent responses"},
                timestamp=datetime.now()
            )
        
        # If only one response, return it
        if len(responses) == 1:
            self.logger.debug("Single response, returning as is")
            return responses[0]
        
        # Combine responses based on query type
        if query_type == QueryType.VISUAL:
            # For visual queries, prioritize visual analyst's response
            visual_response = next(
                (r for r in responses if r.metadata["agent"] == "visual_analyst"),
                None
            )
            if visual_response:
                self.logger.debug("Using visual analyst's response as primary")
                return visual_response
        
        # Combine responses with proper attribution
        self.logger.debug("Combining multiple agent responses")
        combined_content = []
        for response in responses:
            agent_name = response.metadata["agent"].replace("_", " ").title()
            if response.metadata["role"] == "primary":
                combined_content.insert(0, response.content)
            else:
                combined_content.append(f"\nAdditional insights from {agent_name}:\n{response.content}")
        
        synthesized = AgentResponse(
            content="\n".join(combined_content),
            confidence=max(r.confidence for r in responses),
            metadata={
                "combined_from": [r.metadata["agent"] for r in responses],
                "query_type": query_type.value
            },
            timestamp=datetime.now()
        )
        
        self.logger.debug("Response synthesis completed")
        return synthesized

    def _update_context(self, query: str, query_type: QueryType):
        """Update conversation context."""
        self.logger.debug("Updating conversation context")
        self.conversation_context.append({
            "query": query,
            "type": query_type.value,
            "timestamp": datetime.now().isoformat()
        })
        
        # Maintain context window
        if len(self.conversation_context) > self.max_context_length:
            self.logger.debug("Trimming conversation context to maintain max length")
            self.conversation_context = self.conversation_context[-self.max_context_length:]

    def _update_metrics(self, query_type: QueryType, responses: List[AgentResponse]):
        """Update agent performance metrics."""
        self.logger.debug("Updating agent performance metrics")
        for response in responses:
            agent_name = response.metadata["agent"]
            if agent_name in self.agent_metrics:
                metrics = self.agent_metrics[agent_name]
                metrics["queries_handled"] += 1
                
                # Update success rate (based on confidence)
                success_rate = metrics["success_rate"]
                queries = metrics["queries_handled"]
                metrics["success_rate"] = (success_rate * (queries - 1) + response.confidence) / queries
                
                self.logger.debug(f"Updated metrics for agent {agent_name}: {metrics}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        self.logger.info("Retrieving agent performance metrics")
        metrics = {
            "agent_metrics": self.agent_metrics,
            "context_length": len(self.conversation_context)
        }
        self.logger.debug(f"Current metrics: {json.dumps(metrics, indent=2)}")
        return metrics

