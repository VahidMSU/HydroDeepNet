import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
from collections import defaultdict
import spacy
import json
from pathlib import Path

from Logger import Logger

@dataclass
class Concept:
    """Represents a concept node in the knowledge graph."""
    name: str
    domain: str
    description: Optional[str] = None
    attributes: Dict[str, Any] = None
    sources: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class Relationship:
    """Represents a relationship between concepts."""
    source: str
    target: str
    relation_type: str
    weight: float
    metadata: Dict[str, Any] = None
    created_at: datetime = None

class KnowledgeGraph:
    """Manages semantic relationships and domain knowledge."""
    
    def __init__(self, 
                 spacy_model: str = "en_core_web_sm",
                 log_dir: str = "logs",
                 log_level: int = logging.INFO):
        """
        Initialize the knowledge graph.
        
        Args:
            spacy_model: Name of the spaCy model to use for NLP
            log_dir: Directory for log files
            log_level: Logging level
        """
        # Set up logging
        self.logger = Logger(
            log_dir=log_dir,
            app_name="knowledge_graph",
            log_level=log_level
        )
        
        self.logger.info("Initializing KnowledgeGraph")
        
        # Initialize graph structure
        self.graph = nx.DiGraph()
        
        # Load spaCy model for NLP
        try:
            self.logger.info(f"Loading spaCy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)
            self.logger.info(f"Successfully loaded spaCy model")
        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {str(e)}", exc_info=True)
            self.nlp = None
        
        # Initialize domain knowledge
        self.domain_terms = {
            "hydrology": {
                "terms": [
                    "groundwater", "aquifer", "water level", "flow", "recharge", "discharge",
                    "streamflow", "runoff", "infiltration", "hydraulic conductivity",
                    "water table", "piezometric", "porosity", "permeability", "transmissivity"
                ],
                "relationships": {
                    "affects": ["water level", "flow", "discharge"],
                    "measures": ["hydraulic conductivity", "porosity", "permeability"],
                    "contains": ["water table", "aquifer"]
                }
            },
            "climate": {
                "terms": [
                    "precipitation", "rainfall", "temperature", "evaporation", "evapotranspiration",
                    "drought", "flood", "seasonal", "annual", "monthly", "humidity", "wind",
                    "climate change", "weather", "storm", "extreme events"
                ],
                "relationships": {
                    "influences": ["evaporation", "runoff", "infiltration"],
                    "causes": ["flood", "drought"],
                    "measures": ["rainfall", "temperature", "humidity"]
                }
            },
            "agriculture": {
                "terms": [
                    "crop", "yield", "irrigation", "soil", "land use", "vegetation",
                    "fertilizer", "NDVI", "ET", "land cover", "planting", "harvesting",
                    "growing season", "crop rotation", "agricultural practices"
                ],
                "relationships": {
                    "requires": ["water", "soil", "fertilizer"],
                    "produces": ["yield", "vegetation"],
                    "affects": ["land use", "soil quality"]
                }
            }
        }
        
        # Initialize the graph with domain knowledge
        self._initialize_graph()
        
        self.logger.info("KnowledgeGraph initialization completed")
    
    def _initialize_graph(self):
        """Initialize the graph with domain knowledge."""
        self.logger.info("Initializing graph with domain knowledge")
        try:
            # Add concepts for each domain
            concepts_added = 0
            relationships_added = 0
            
            for domain, data in self.domain_terms.items():
                self.logger.debug(f"Adding terms for domain: {domain}")
                for term in data["terms"]:
                    self.add_concept(
                        name=term,
                        domain=domain,
                        attributes={"type": "domain_term"}
                    )
                    concepts_added += 1
                
                # Add relationships
                self.logger.debug(f"Adding relationships for domain: {domain}")
                for relation_type, targets in data["relationships"].items():
                    for target in targets:
                        if target in data["terms"]:
                            self.add_relationship(
                                source=domain,
                                target=target,
                                relation_type=relation_type,
                                weight=1.0
                            )
                            relationships_added += 1
            
            self.logger.info(f"Successfully initialized knowledge graph with {concepts_added} concepts and {relationships_added} relationships")
            
        except Exception as e:
            self.logger.error(f"Error initializing graph: {str(e)}", exc_info=True)
    
    def add_concept(self, 
                   name: str, 
                   domain: str, 
                   description: Optional[str] = None,
                   attributes: Optional[Dict[str, Any]] = None,
                   sources: Optional[List[str]] = None) -> bool:
        """
        Add a concept to the knowledge graph.
        
        Args:
            name: Name of the concept
            domain: Domain the concept belongs to
            description: Optional description
            attributes: Optional attributes
            sources: Optional list of source documents
            
        Returns:
            bool: Success status
        """
        self.logger.debug(f"Adding concept: {name} in domain: {domain}")
        try:
            concept = Concept(
                name=name,
                domain=domain,
                description=description,
                attributes=attributes or {},
                sources=sources or [],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.graph.add_node(
                name,
                **concept.__dict__
            )
            
            self.logger.debug(f"Successfully added concept: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding concept: {str(e)}", exc_info=True)
            return False
    
    def add_relationship(self,
                        source: str,
                        target: str,
                        relation_type: str,
                        weight: float = 1.0,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a relationship between concepts.
        
        Args:
            source: Source concept name
            target: Target concept name
            relation_type: Type of relationship
            weight: Relationship weight
            metadata: Optional metadata
            
        Returns:
            bool: Success status
        """
        self.logger.debug(f"Adding relationship: {source} -{relation_type}-> {target}")
        try:
            relationship = Relationship(
                source=source,
                target=target,
                relation_type=relation_type,
                weight=weight,
                metadata=metadata or {},
                created_at=datetime.now()
            )
            
            self.graph.add_edge(
                source,
                target,
                **relationship.__dict__
            )
            
            self.logger.debug(f"Successfully added relationship: {source} -{relation_type}-> {target}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding relationship: {str(e)}", exc_info=True)
            return False
    
    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract concepts from text using NLP.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted concepts with metadata
        """
        self.logger.debug(f"Extracting concepts from text: {text[:50]}...")
        if not self.nlp:
            self.logger.warning("NLP model not available, can't extract concepts")
            return []
            
        try:
            doc = self.nlp(text)
            concepts = []
            
            # Extract named entities
            for ent in doc.ents:
                concepts.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                concepts.append({
                    "text": chunk.text,
                    "type": "NOUN_PHRASE",
                    "start": chunk.start_char,
                    "end": chunk.end_char
                })
            
            self.logger.debug(f"Extracted {len(concepts)} concepts")
            return concepts
            
        except Exception as e:
            self.logger.error(f"Error extracting concepts: {str(e)}", exc_info=True)
            return []
    
    def find_related_concepts(self, 
                            concept: str, 
                            max_distance: int = 2) -> List[Dict[str, Any]]:
        """
        Find concepts related to the given concept.
        
        Args:
            concept: Concept to find relations for
            max_distance: Maximum path length to consider
            
        Returns:
            List of related concepts with relationship info
        """
        self.logger.debug(f"Finding concepts related to: {concept}, max_distance: {max_distance}")
        try:
            if concept not in self.graph:
                self.logger.debug(f"Concept not found in graph: {concept}")
                return []
            
            related = []
            
            # Find paths up to max_distance
            for node in nx.single_source_shortest_path_length(
                self.graph, concept, cutoff=max_distance
            ).items():
                target, distance = node
                if target != concept:
                    # Get relationship information
                    path = nx.shortest_path(self.graph, concept, target)
                    edges = list(zip(path[:-1], path[1:]))
                    relationships = [
                        self.graph.edges[edge]["relation_type"]
                        for edge in edges
                    ]
                    
                    related.append({
                        "concept": target,
                        "distance": distance,
                        "path": path,
                        "relationships": relationships
                    })
            
            self.logger.debug(f"Found {len(related)} related concepts")
            return related
            
        except Exception as e:
            self.logger.error(f"Error finding related concepts: {str(e)}", exc_info=True)
            return []
    
    def answer_query(self, query: str) -> Optional[str]:
        """
        Answer a query using the knowledge graph.
        
        Args:
            query: Query text
            
        Returns:
            Response text or None if no answer found
        """
        self.logger.info(f"Answering query: {query}")
        try:
            # Extract concepts from query
            concepts = self.extract_concepts(query)
            if not concepts:
                self.logger.debug("No concepts extracted from query")
                return None
            
            # Find relevant concepts in graph
            relevant_concepts = []
            for concept in concepts:
                concept_name = concept["text"].lower()
                if concept_name in self.graph:
                    relevant_concepts.append(concept_name)
            
            if not relevant_concepts:
                self.logger.debug("No relevant concepts found in graph")
                return None
            
            # Get related concepts and build response
            response_parts = []
            for concept in relevant_concepts:
                related = self.find_related_concepts(concept)
                if related:
                    response_parts.append(f"Information about {concept}:")
                    for rel in related:
                        path_desc = " -> ".join(rel["relationships"])
                        response_parts.append(
                            f"- Related to {rel['concept']} through: {path_desc}"
                        )
            
            if response_parts:
                self.logger.info("Successfully generated response from knowledge graph")
                return "\n".join(response_parts)
                
            self.logger.debug("No response parts generated")
            return None
            
        except Exception as e:
            self.logger.error(f"Error answering query: {str(e)}", exc_info=True)
            return None
    
    def update_from_interaction(self, 
                              query: str, 
                              response: str,
                              metadata: Optional[Dict[str, Any]] = None):
        """
        Update knowledge graph based on interaction.
        
        Args:
            query: User query
            response: System response
            metadata: Optional interaction metadata
        """
        self.logger.info("Updating knowledge graph from interaction")
        try:
            # Extract concepts from both query and response
            query_concepts = self.extract_concepts(query)
            response_concepts = self.extract_concepts(response)
            
            concepts_added = 0
            relationships_added = 0
            
            # Add new concepts and relationships
            for concept in query_concepts + response_concepts:
                concept_name = concept["text"].lower()
                if concept_name not in self.graph:
                    self.add_concept(
                        name=concept_name,
                        domain="user_interaction",
                        attributes={"type": concept["type"]},
                        sources=[f"interaction_{datetime.now().isoformat()}"]
                    )
                    concepts_added += 1
            
            # Add relationships between concepts that appear together
            for q_concept in query_concepts:
                for r_concept in response_concepts:
                    q_name = q_concept["text"].lower()
                    r_name = r_concept["text"].lower()
                    if q_name != r_name:  # Don't create self-relationships
                        self.add_relationship(
                            source=q_name,
                            target=r_name,
                            relation_type="appears_with",
                            weight=1.0,
                            metadata=metadata
                        )
                        relationships_added += 1
            
            self.logger.info(f"Updated graph with {concepts_added} new concepts and {relationships_added} new relationships")
            
        except Exception as e:
            self.logger.error(f"Error updating from interaction: {str(e)}", exc_info=True)
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save the knowledge graph to a file.
        
        Args:
            file_path: Path to save the graph
            
        Returns:
            bool: Success status
        """
        self.logger.info(f"Saving knowledge graph to: {file_path}")
        try:
            # Convert graph to dictionary
            graph_data = nx.node_link_data(self.graph)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            self.logger.info(f"Successfully saved knowledge graph to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving graph: {str(e)}", exc_info=True)
            return False
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Load the knowledge graph from a file.
        
        Args:
            file_path: Path to load the graph from
            
        Returns:
            bool: Success status
        """
        self.logger.info(f"Loading knowledge graph from: {file_path}")
        try:
            # Load from file
            with open(file_path, 'r') as f:
                graph_data = json.load(f)
            
            # Convert dictionary to graph
            self.graph = nx.node_link_graph(graph_data)
            
            nodes_count = self.graph.number_of_nodes()
            edges_count = self.graph.number_of_edges()
            
            self.logger.info(f"Successfully loaded knowledge graph with {nodes_count} nodes and {edges_count} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading graph: {str(e)}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        self.logger.info("Gathering knowledge graph statistics")
        try:
            # Calculate statistics
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            
            stats = {
                "num_concepts": num_nodes,
                "num_relationships": num_edges,
                "num_domains": len(set(
                    data["domain"] for _, data in self.graph.nodes(data=True)
                )),
                "density": nx.density(self.graph),
                "avg_degree": sum(dict(self.graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
            }
            
            self.logger.debug(f"Graph statistics: {json.dumps(stats, indent=2)}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}", exc_info=True)
            return {}

