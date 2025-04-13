import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import os
from pathlib import Path
import re
from datetime import datetime
import spacy

# Import the config loader
from config_loader import get_config

class KnowledgeGraph:
    """
    Manages a graph of concepts and relationships extracted from report data.
    This helps the agent understand connections between different environmental factors.
    """
    
    def __init__(self,
                 storage_path: Optional[str] = None,
                 use_spacy: bool = True,
                 spacy_model: Optional[str] = None,
                 logger=None):
        """
        Initialize the knowledge graph.
        
        Args:
            storage_path: Path to save the graph
            use_spacy: Whether to use spaCy for entity extraction
            spacy_model: spaCy model to use
            logger: Logger instance to use
        """
        # Get config values
        config = get_config()
        kg_path_config = config.get('knowledge_graph_json', 'knowledge_graph.json')
        spacy_model_config = config.get('spacy_model', 'en_core_web_sm')

        self.storage_path = storage_path or kg_path_config
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model or spacy_model_config
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize the graph
        self.graph = nx.DiGraph()
        
        # Ensure the directory for the graph exists if it's not in the current dir
        graph_dir = Path(self.storage_path).parent
        if graph_dir != Path('.'):
            os.makedirs(graph_dir, exist_ok=True)
        
        # Load spaCy model if requested
        self.nlp = None
        if self.use_spacy:
            try:
                self.nlp = spacy.load(self.spacy_model)
                self.logger.info(f"Loaded spaCy model: {self.spacy_model}")
            except Exception as e:
                self.logger.warning(f"Could not load spaCy model '{self.spacy_model}': {str(e)}. Disabling spaCy features.")
                self.use_spacy = False
        
        # Define domains and their related terms
        self.domains = {
            "climate": [
                "temperature", "precipitation", "rainfall", "drought", "humidity",
                "weather", "climate change", "global warming"
            ],
            "groundwater": [
                "aquifer", "water table", "water level", "drawdown", "recharge",
                "well", "pumping", "hydraulic conductivity", "transmissivity"
            ],
            "agriculture": [
                "crop", "farming", "irrigation", "yield", "harvest",
                "corn", "soybean", "wheat", "rotation"
            ],
            "soil": [
                "fertility", "texture", "erosion", "drainage", "infiltration",
                "organic matter", "compaction", "moisture"
            ],
            "hydrology": [
                "runoff", "streamflow", "river", "discharge", "flood",
                "watershed", "basin", "flow", "evapotranspiration"
            ]
        }
        
        # Domain-specific relationships
        self.domain_relationships = {
            "climate_groundwater": [
                ("precipitation", "recharge", "increases"),
                ("temperature", "evapotranspiration", "increases"),
                ("drought", "water level", "decreases"),
                ("rainfall", "water table", "raises")
            ],
            "agriculture_soil": [
                ("crop", "soil fertility", "affects"),
                ("irrigation", "soil moisture", "increases"),
                ("crop rotation", "soil fertility", "improves"),
                ("farming", "erosion", "can cause")
            ],
            "groundwater_agriculture": [
                ("water level", "irrigation", "limits"),
                ("aquifer", "pumping", "supplies"),
                ("water table", "crop yield", "affects"),
                ("groundwater", "agriculture", "supports")
            ],
            "climate_soil": [
                ("precipitation", "soil moisture", "increases"),
                ("temperature", "soil moisture", "decreases"),
                ("rainfall", "erosion", "can cause"),
                ("drought", "soil moisture", "decreases")
            ]
        }
        
        # Load existing graph if available
        self.load_graph()
    
    def load_graph(self) -> bool:
        """
        Load the knowledge graph from storage.
        
        Returns:
            Success status
        """
        if not os.path.exists(self.storage_path):
            self.logger.info("No existing knowledge graph found. Creating new graph.")
            return False
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            # Create a new graph
            G = nx.DiGraph()
            
            # Add nodes
            for node_id, node_data in data.get("nodes", {}).items():
                G.add_node(node_id, **node_data)
                
            # Add edges
            for edge in data.get("edges", []):
                G.add_edge(
                    edge["source"], 
                    edge["target"], 
                    type=edge["type"],
                    weight=edge.get("weight", 1.0),
                    **edge.get("attributes", {})
                )
                
            self.graph = G
            self.logger.info(f"Loaded knowledge graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {str(e)}")
            return False
    
    def save_graph(self) -> bool:
        """
        Save the knowledge graph to storage.
        
        Returns:
            Success status
        """
        try:
            # Convert graph to serializable format
            data = {
                "nodes": {},
                "edges": []
            }
            
            # Add nodes
            for node_id, node_data in self.graph.nodes(data=True):
                data["nodes"][node_id] = node_data
                
            # Add edges
            for source, target, edge_data in self.graph.edges(data=True):
                edge_info = {
                    "source": source,
                    "target": target,
                    "type": edge_data.get("type", "related_to"),
                    "weight": edge_data.get("weight", 1.0)
                }
                
                # Add other attributes
                attributes = {k: v for k, v in edge_data.items() if k not in ["type", "weight"]}
                if attributes:
                    edge_info["attributes"] = attributes
                    
                data["edges"].append(edge_info)
                
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved knowledge graph to {self.storage_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {str(e)}")
            return False
    
    def add_concept(self, name: str, domain: str, attributes: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a concept to the graph.
        
        Args:
            name: Concept name
            domain: Domain the concept belongs to
            attributes: Additional attributes
            
        Returns:
            Success status
        """
        try:
            # Normalize concept name
            name = name.lower().strip()
            
            # Check if node already exists
            if name in self.graph.nodes:
                # Update domain if not already set
                if "domain" not in self.graph.nodes[name]:
                    self.graph.nodes[name]["domain"] = domain
                    
                # Update attributes
                if attributes:
                    for key, value in attributes.items():
                        self.graph.nodes[name][key] = value
                        
                self.logger.debug(f"Updated concept: {name}")
            else:
                # Add new node
                node_data = {
                    "domain": domain,
                    "added_at": datetime.now().isoformat()
                }
                
                if attributes:
                    node_data.update(attributes)
                    
                self.graph.add_node(name, **node_data)
                self.logger.debug(f"Added new concept: {name}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding concept: {str(e)}")
            return False
    
    def add_relationship(self, 
                          source: str, 
                          target: str, 
                          relation_type: str = "related_to", 
                          weight: float = 1.0,
                          attributes: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a relationship between concepts.
        
        Args:
            source: Source concept
            target: Target concept
            relation_type: Type of relationship
            weight: Relationship strength
            attributes: Additional attributes
            
        Returns:
            Success status
        """
        try:
            # Normalize concept names
            source = source.lower().strip()
            target = target.lower().strip()
            
            # Check if nodes exist, add them if not
            if source not in self.graph.nodes:
                self.add_concept(source, "unknown")
                
            if target not in self.graph.nodes:
                self.add_concept(target, "unknown")
                
            # Check if edge already exists
            if self.graph.has_edge(source, target):
                # Update weight (increase it slightly)
                current_weight = self.graph.edges[source, target].get("weight", 1.0)
                self.graph.edges[source, target]["weight"] = current_weight + 0.1
                
                # Update relation type if not already set
                if "type" not in self.graph.edges[source, target]:
                    self.graph.edges[source, target]["type"] = relation_type
                    
                # Update attributes
                if attributes:
                    for key, value in attributes.items():
                        self.graph.edges[source, target][key] = value
                        
                self.logger.debug(f"Updated relationship: {source} -> {target}")
            else:
                # Add new edge
                edge_data = {
                    "type": relation_type,
                    "weight": weight,
                    "added_at": datetime.now().isoformat()
                }
                
                if attributes:
                    edge_data.update(attributes)
                    
                self.graph.add_edge(source, target, **edge_data)
                self.logger.debug(f"Added new relationship: {source} -> {target}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding relationship: {str(e)}")
            return False
    
    def initialize_domain_concepts(self) -> int:
        """
        Initialize the graph with domain concepts.
        
        Returns:
            Number of concepts added
        """
        count = 0
        
        try:
            # Add domains as concepts
            for domain, terms in self.domains.items():
                self.add_concept(domain, domain, {"is_domain": True})
                count += 1
                
                # Add terms under each domain
                for term in terms:
                    self.add_concept(term, domain)
                    self.add_relationship(domain, term, "includes")
                    count += 1
                    
            # Add cross-domain relationships
            for relation_group, relationships in self.domain_relationships.items():
                for source, target, relation_type in relationships:
                    self.add_relationship(source, target, relation_type)
                    count += 1
                    
            self.logger.info(f"Initialized knowledge graph with {count} concepts and relationships")
            return count
            
        except Exception as e:
            self.logger.error(f"Error initializing domain concepts: {str(e)}")
            return count
    
    def extract_concepts_from_text(self, text: str, domain: Optional[str] = None) -> List[str]:
        """
        Extract concepts from text.
        
        Args:
            text: Text to extract concepts from
            domain: Optional domain to filter concepts
            
        Returns:
            List of extracted concepts
        """
        concepts = []
        
        # Use spaCy for extraction if available
        if self.use_spacy and self.nlp:
            try:
                doc = self.nlp(text)
                
                # Extract entities and noun phrases
                for ent in doc.ents:
                    if ent.label_ in ["ORG", "GPE", "LOC", "PRODUCT", "EVENT"]:
                        concepts.append(ent.text.lower())
                
                # Extract noun chunks (potential concepts)
                for chunk in doc.noun_chunks:
                    concepts.append(chunk.text.lower())
                    
            except Exception as e:
                self.logger.warning(f"Error extracting concepts with spaCy: {str(e)}")
        
        # Simple keyword matching as fallback or supplement
        for domain_name, terms in self.domains.items():
            if domain and domain != domain_name:
                continue
                
            for term in terms:
                if re.search(r'\b' + re.escape(term.lower()) + r'\b', text.lower()):
                    concepts.append(term.lower())
        
        # Remove duplicates and clean up
        concepts = list(set(concepts))
        concepts = [c.strip() for c in concepts if len(c.strip()) > 2]
        
        return concepts
    
    def process_text(self, text: str, source: str, domain: Optional[str] = None) -> int:
        """
        Process text to extract concepts and relationships.
        
        Args:
            text: Text to process
            source: Source of the text (for attribution)
            domain: Optional domain to associate with concepts
            
        Returns:
            Number of concepts and relationships added
        """
        count = 0
        try:
            # Extract concepts
            concepts = self.extract_concepts_from_text(text, domain)
            
            # Add concepts to graph
            for concept in concepts:
                if self.add_concept(concept, domain or "unknown", {"source": source}):
                    count += 1
            
            # Add co-occurrence relationships
            for i in range(len(concepts)):
                for j in range(i+1, len(concepts)):
                    if self.add_relationship(
                        concepts[i], 
                        concepts[j], 
                        "co_occurs_with", 
                        weight=0.5,
                        attributes={"source": source}
                    ):
                        count += 1
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return count
    
    def get_related_concepts(self, concept: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get concepts related to the given concept.
        
        Args:
            concept: Concept name
            max_depth: Maximum relationship depth
            
        Returns:
            Dictionary of related concepts and relationship info
        """
        concept = concept.lower().strip()
        
        if concept not in self.graph.nodes:
            return {"concept": concept, "exists": False, "related": []}
            
        related = {}
        
        # Get direct relationships (depth 1)
        for neighbor in self.graph.successors(concept):
            edge_data = self.graph.edges[concept, neighbor]
            related[neighbor] = {
                "relationship": edge_data.get("type", "related_to"),
                "weight": edge_data.get("weight", 1.0),
                "depth": 1
            }
            
        # Get deeper relationships if requested
        if max_depth > 1:
            depth_1_neighbors = list(related.keys())
            
            for depth in range(2, max_depth + 1):
                for neighbor in depth_1_neighbors:
                    for secondary_neighbor in self.graph.successors(neighbor):
                        if secondary_neighbor not in related and secondary_neighbor != concept:
                            edge_data = self.graph.edges[neighbor, secondary_neighbor]
                            related[secondary_neighbor] = {
                                "relationship": edge_data.get("type", "related_to"),
                                "via": neighbor,
                                "weight": edge_data.get("weight", 1.0),
                                "depth": depth
                            }
        
        # Convert to list format
        result = []
        for related_concept, data in related.items():
            item = {
                "concept": related_concept,
                "domain": self.graph.nodes[related_concept].get("domain", "unknown"),
                **data
            }
            result.append(item)
            
        # Sort by weight and depth
        result.sort(key=lambda x: (x["depth"], -x["weight"]))
        
        return {
            "concept": concept,
            "exists": True,
            "domain": self.graph.nodes[concept].get("domain", "unknown"),
            "related": result
        }
    
    def find_path(self, source: str, target: str) -> List[Dict[str, Any]]:
        """
        Find a path between two concepts.
        
        Args:
            source: Source concept
            target: Target concept
            
        Returns:
            List of nodes and edges in the path
        """
        source = source.lower().strip()
        target = target.lower().strip()
        
        if source not in self.graph.nodes or target not in self.graph.nodes:
            return []
            
        try:
            # Find shortest path
            path_nodes = nx.shortest_path(self.graph, source=source, target=target)
            
            result = []
            for i in range(len(path_nodes) - 1):
                current = path_nodes[i]
                next_node = path_nodes[i + 1]
                edge_data = self.graph.edges[current, next_node]
                
                result.append({
                    "source": current,
                    "source_domain": self.graph.nodes[current].get("domain", "unknown"),
                    "target": next_node,
                    "target_domain": self.graph.nodes[next_node].get("domain", "unknown"),
                    "relationship": edge_data.get("type", "related_to"),
                    "weight": edge_data.get("weight", 1.0)
                })
                
            return result
            
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            self.logger.error(f"Error finding path: {str(e)}")
            return []
    
    def generate_question(self, concept: str, max_depth: int = 2) -> str:
        """
        Generate a question based on the concept and its relationships.
        
        Args:
            concept: Concept to generate a question about
            max_depth: Maximum relationship depth
            
        Returns:
            Generated question
        """
        concept = concept.lower().strip()
        
        if concept not in self.graph.nodes:
            return f"What is {concept} and how does it relate to environmental factors?"
            
        # Get related concepts
        related_data = self.get_related_concepts(concept, max_depth)
        
        if not related_data["related"]:
            return f"What is {concept} and how does it relate to environmental factors?"
            
        # Pick the most strongly related concept
        strongest_relation = related_data["related"][0]
        related_concept = strongest_relation["concept"]
        relation_type = strongest_relation.get("relationship", "related_to")
        
        # Generate question templates based on relationship type
        templates = {
            "increases": [
                f"How does {concept} increase {related_concept}?",
                f"What is the relationship between {concept} and {related_concept}?",
                f"Why does {concept} lead to higher {related_concept}?"
            ],
            "decreases": [
                f"How does {concept} decrease {related_concept}?",
                f"What is the inverse relationship between {concept} and {related_concept}?",
                f"Why does {concept} lead to lower {related_concept}?"
            ],
            "affects": [
                f"How does {concept} affect {related_concept}?",
                f"What is the impact of {concept} on {related_concept}?",
                f"What happens to {related_concept} when {concept} changes?"
            ],
            "includes": [
                f"What aspects of {related_concept} are part of {concept}?",
                f"How is {related_concept} included in {concept}?",
                f"What is the relationship between {concept} and its component {related_concept}?"
            ],
            "co_occurs_with": [
                f"Why do {concept} and {related_concept} occur together?",
                f"What is the relationship between {concept} and {related_concept}?",
                f"How are {concept} and {related_concept} connected?"
            ],
            "related_to": [
                f"How are {concept} and {related_concept} related?",
                f"What is the connection between {concept} and {related_concept}?",
                f"How does {concept} influence {related_concept}?"
            ]
        }
        
        # Default to related_to if relation_type not in templates
        template_list = templates.get(relation_type, templates["related_to"])
        
        # Pick a template based on the concept's position in the graph
        import random
        selected_template = random.choice(template_list)
        
        return selected_template
    
    def visualize(self, output_path: str = "knowledge_graph.png", max_nodes: int = 50) -> bool:
        """
        Visualize the knowledge graph.
        
        Args:
            output_path: Path to save the visualization
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Success status
        """
        try:
            # Create a subgraph if the graph is too large
            if len(self.graph.nodes) > max_nodes:
                # Get the nodes with the highest degree
                node_degrees = sorted(
                    [(n, self.graph.degree(n)) for n in self.graph.nodes],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                top_nodes = [n for n, d in node_degrees[:max_nodes]]
                subgraph = self.graph.subgraph(top_nodes)
            else:
                subgraph = self.graph
                
            # Create a spring layout
            pos = nx.spring_layout(subgraph, seed=42)
            
            # Get domain colors
            domain_colors = {
                "climate": "red",
                "groundwater": "blue",
                "agriculture": "green",
                "soil": "brown",
                "hydrology": "purple",
                "unknown": "gray"
            }
            
            # Get node colors based on domains
            node_colors = [
                domain_colors.get(subgraph.nodes[n].get("domain", "unknown"), "gray") 
                for n in subgraph.nodes
            ]
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Draw nodes
            nx.draw_networkx_nodes(
                subgraph, 
                pos, 
                node_color=node_colors,
                node_size=500,
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                subgraph,
                pos,
                width=[subgraph.edges[e].get("weight", 1.0) for e in subgraph.edges],
                alpha=0.5,
                arrows=True
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                subgraph,
                pos,
                font_size=10,
                font_family="sans-serif"
            )
            
            # Save figure
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            self.logger.info(f"Saved visualization to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary of statistics
        """
        try:
            # Get basic stats
            num_nodes = len(self.graph.nodes)
            num_edges = len(self.graph.edges)
            
            # Get domain distribution
            domains = {}
            for node, data in self.graph.nodes(data=True):
                domain = data.get("domain", "unknown")
                domains[domain] = domains.get(domain, 0) + 1
                
            # Get relationship type distribution
            relationships = {}
            for u, v, data in self.graph.edges(data=True):
                rel_type = data.get("type", "related_to")
                relationships[rel_type] = relationships.get(rel_type, 0) + 1
                
            # Get central concepts (highest degree)
            central_concepts = sorted(
                [(n, self.graph.degree(n)) for n in self.graph.nodes],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                "nodes": num_nodes,
                "edges": num_edges,
                "domain_distribution": domains,
                "relationship_types": relationships,
                "central_concepts": central_concepts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting graph stats: {str(e)}")
            return {
                "nodes": 0,
                "edges": 0,
                "error": str(e)
            }
