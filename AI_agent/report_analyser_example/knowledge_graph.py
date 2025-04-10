"""
Knowledge graph utilities for extracting and connecting concepts from various document types.
"""
import os
import pandas as pd
import logging
from prompt_handler import extract_keywords

class KnowledgeGraph:
    """
    A class for creating and managing a knowledge graph of concepts and relationships.
    
    This class extracts concepts from various file types (markdown, CSV, images)
    and builds relationships between them based on co-occurrence patterns.
    """
    
    def __init__(self, discovered_files=None, logger=None):
        """
        Initialize the KnowledgeGraph.
        
        Args:
            discovered_files: Dictionary of discovered files by type
            logger: Optional logger instance for logging
        """
        self.discovered_files = discovered_files or {}
        self.logger = logger
        self.knowledge_graph = None
        
    def create_knowledge_graph(self):
        """
        Create and populate a knowledge graph to represent domain concepts and their relationships.
        
        Returns:
            bool: Success status
        """
        if self.logger:
            self.logger.info("Creating knowledge graph")
        
        # Initialize knowledge graph structure
        self.knowledge_graph = {
            "nodes": {},  # Concepts and entities
            "relationships": {},  # How concepts are connected
            "file_concepts": {},  # Mapping from files to concepts they contain
            "domain_terms": {
                "hydrology": ["groundwater", "aquifer", "water level", "flow", "recharge", "discharge", 
                              "streamflow", "runoff", "infiltration", "hydraulic conductivity"],
                "climate": ["precipitation", "rainfall", "temperature", "evaporation", "evapotranspiration",
                           "drought", "flood", "seasonal", "annual", "monthly"],
                "agriculture": ["crop", "yield", "irrigation", "soil", "land use", "vegetation", 
                               "fertilizer", "NDVI", "ET", "land cover"],
                "geography": ["location", "region", "spatial", "map", "terrain", "elevation", "slope", 
                             "aspect", "watershed", "basin"],
                "time": ["temporal", "time series", "seasonal", "annual", "trend", "change", "variability",
                        "extreme", "prediction", "forecast", "simulation"]
            }
        }
        
        try:
            # Extract concepts from markdown files
            self._populate_knowledge_from_docs()
            
            # Extract concepts from CSV headers and data
            self._populate_knowledge_from_csv()
            
            # Extract concepts from image filenames
            self._populate_knowledge_from_images()
            
            # Build relationships between concepts
            self._build_concept_relationships()
            
            if self.logger:
                self.logger.info(f"Knowledge graph created with {len(self.knowledge_graph['nodes'])} concepts and {len(self.knowledge_graph['relationships'])} relationships")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"create_knowledge_graph Error creating knowledge graph: {str(e)}")
            return False
            
    def _populate_knowledge_from_docs(self):
        """Extract concepts and relationships from markdown documentation."""
        md_files = self.discovered_files.get('md', [])
        
        for md_path in md_files:
            file_name = os.path.basename(md_path)
            self.knowledge_graph["file_concepts"][file_name] = set()
            
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract all domain terms that appear in the document
                for domain, terms in self.knowledge_graph["domain_terms"].items():
                    for term in terms:
                        if term.lower() in content.lower():
                            # Add or update node for this term
                            if term not in self.knowledge_graph["nodes"]:
                                self.knowledge_graph["nodes"][term] = {
                                    "domain": domain,
                                    "count": 1,
                                    "files": [file_name]
                                }
                            else:
                                self.knowledge_graph["nodes"][term]["count"] += 1
                                if file_name not in self.knowledge_graph["nodes"][term]["files"]:
                                    self.knowledge_graph["nodes"][term]["files"].append(file_name)
                            
                            # Add to file concepts
                            self.knowledge_graph["file_concepts"][file_name].add(term)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"_populate_knowledge_from_docs Error extracting knowledge from {file_name}: {str(e)}")
                
    def _populate_knowledge_from_csv(self):
        """Extract concepts from CSV headers and data."""
        csv_files = self.discovered_files.get('csv', [])
        
        for csv_path in csv_files:
            file_name = os.path.basename(csv_path)
            self.knowledge_graph["file_concepts"][file_name] = set()
            
            try:
                # Check file size first before reading
                row_count = 0
                with open(csv_path, 'r') as f:
                    for i, _ in enumerate(f):
                        row_count = i + 1
                        if row_count > 1000:
                            break
                
                # Read the file (first 100 rows only if large)
                if row_count > 1000:
                    df = pd.read_csv(csv_path, nrows=100)
                else:
                    df = pd.read_csv(csv_path)
                
                # Extract concepts from column names
                for column in df.columns:
                    column_lower = column.lower()
                    
                    # Check against domain terms
                    for domain, terms in self.knowledge_graph["domain_terms"].items():
                        for term in terms:
                            if term.lower() in column_lower:
                                # Add or update node for this term
                                if term not in self.knowledge_graph["nodes"]:
                                    self.knowledge_graph["nodes"][term] = {
                                        "domain": domain,
                                        "count": 1,
                                        "files": [file_name]
                                    }
                                else:
                                    self.knowledge_graph["nodes"][term]["count"] += 1
                                    if file_name not in self.knowledge_graph["nodes"][term]["files"]:
                                        self.knowledge_graph["nodes"][term]["files"].append(file_name)
                                
                                # Add to file concepts
                                self.knowledge_graph["file_concepts"][file_name].add(term)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"_populate_knowledge_from_csv Error extracting knowledge from {file_name}: {str(e)}")
                

    def _populate_knowledge_from_images(self):
        """Extract concepts from image filenames."""
        for img_type in ['png', 'jpg']:
            for img_path in self.discovered_files.get(img_type, []):
                file_name = os.path.basename(img_path)
                self.knowledge_graph["file_concepts"][file_name] = set()
                
                # Extract concepts from filename
                file_name_lower = file_name.lower()
                
                # Check against domain terms
                for domain, terms in self.knowledge_graph["domain_terms"].items():
                    for term in terms:
                        if term.lower() in file_name_lower:
                            # Add or update node for this term
                            if term not in self.knowledge_graph["nodes"]:
                                self.knowledge_graph["nodes"][term] = {
                                    "domain": domain,
                                    "count": 1,
                                    "files": [file_name]
                                }
                            else:
                                self.knowledge_graph["nodes"][term]["count"] += 1
                                if file_name not in self.knowledge_graph["nodes"][term]["files"]:
                                    self.knowledge_graph["nodes"][term]["files"].append(file_name)
                            
                            # Add to file concepts
                            self.knowledge_graph["file_concepts"][file_name].add(term)
                
    def _build_concept_relationships(self):
        """Build relationships between concepts based on co-occurrence."""
        # Find concepts that co-occur in the same files
        for file_name, concepts in self.knowledge_graph["file_concepts"].items():
            # Create relationships between all pairs of concepts in this file
            concept_list = list(concepts)
            for i in range(len(concept_list)):
                for j in range(i + 1, len(concept_list)):
                    concept1 = concept_list[i]
                    concept2 = concept_list[j]
                    
                    # Create a unique relationship key (alphabetically ordered)
                    rel_key = f"{min(concept1, concept2)}--{max(concept1, concept2)}"
                    
                    if rel_key not in self.knowledge_graph["relationships"]:
                        self.knowledge_graph["relationships"][rel_key] = {
                            "source": min(concept1, concept2),
                            "target": max(concept1, concept2),
                            "weight": 1,
                            "files": [file_name]
                        }
                    else:
                        self.knowledge_graph["relationships"][rel_key]["weight"] += 1
                        if file_name not in self.knowledge_graph["relationships"][rel_key]["files"]:
                            self.knowledge_graph["relationships"][rel_key]["files"].append(file_name)
                            
    def answer_query(self, query):
        """
        Try to answer a query using the knowledge graph.
        
        Args:
            query: The user query text
            
        Returns:
            str: Response based on knowledge graph or None if not applicable
        """
        if not self.knowledge_graph:
            return None
            
        # Extract keywords from the query
        keywords = extract_keywords(query)
        
        # Find concepts in the query
        matched_concepts = []
        for concept in self.knowledge_graph["nodes"]:
            if concept.lower() in query.lower():
                matched_concepts.append(concept)
            
        if not matched_concepts:
            return None
            
        # Find files relevant to these concepts
        relevant_files = {}
        for concept in matched_concepts:
            node = self.knowledge_graph["nodes"].get(concept)
            if node and "files" in node:
                for file in node["files"]:
                    if file not in relevant_files:
                        relevant_files[file] = 1
                    else:
                        relevant_files[file] += 1
        
        # Find relationships between matched concepts
        related_concepts = set()
        for concept in matched_concepts:
            for rel_key, rel in self.knowledge_graph["relationships"].items():
                if rel["source"] == concept or rel["target"] == concept:
                    other_concept = rel["target"] if rel["source"] == concept else rel["source"]
                    related_concepts.add(other_concept)
                    
        # Build a response using the knowledge graph
        if matched_concepts and relevant_files:
            response = f"Based on my knowledge graph, I found information about {', '.join(matched_concepts)} in the following files:\n\n"
            
            # List the most relevant files (files with the most concept matches)
            sorted_files = sorted(relevant_files.items(), key=lambda x: x[1], reverse=True)[:5]
            for file, count in sorted_files:
                response += f"- {file}\n"
                
            # Add information about related concepts
            if related_concepts:
                response += f"\nThese concepts are related to: {', '.join(related_concepts)}.\n"
                response += "Would you like me to analyze any of these files or explore these related concepts further?"
                
            return response
            
        return None
        
    def get_related_files(self, concept):
        """
        Get files related to a specific concept.
        
        Args:
            concept: The concept to look for
            
        Returns:
            list: List of files containing this concept
        """
        if not self.knowledge_graph or concept not in self.knowledge_graph["nodes"]:
            return []
            
        return self.knowledge_graph["nodes"][concept].get("files", []) 