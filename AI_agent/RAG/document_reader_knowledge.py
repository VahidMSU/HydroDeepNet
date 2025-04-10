import os
import re
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union
import networkx as nx
import pandas as pd
import numpy as np
from agno.agent import Agent
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.knowledge.docx import DocxKnowledgeBase

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KnowledgeHandlerLogger")

class KnowledgeHandler:
    """Handles knowledge management and retrieval for documents."""
    
    def __init__(self, document_reader):
        """Initialize the knowledge handler with a reference to the document reader.
        
        Args:
            document_reader: The InteractiveDocumentReader instance
        """
        self.document_reader = document_reader
        self.knowledge_bases = {}
        self.combined_knowledge = None
        self.knowledge_graph = nx.DiGraph()
        self.entity_cache = {}
        # Add markdown-specific cache
        self.markdown_cache = {}
        # Track document relevance for queries
        self.document_relevance_scores = {}
        # Limit search attempts
        self.search_attempts = {}
    
    def initialize_knowledge_base(self, doc_type: str, file_paths: List[str]) -> bool:
        """Initialize a knowledge base for a specific document type.
        
        Args:
            doc_type: Type of document (csv, pdf, text, json, etc.)
            file_paths: List of file paths to include in the knowledge base
            
        Returns:
            Boolean indicating success
        """
        try:
            # Validate file paths
            valid_paths = [path for path in file_paths if os.path.exists(path)]
            
            if not valid_paths:
                logger.warning(f"No valid {doc_type} files provided for knowledge base")
                return False
            
            # Initialize the appropriate knowledge base type
            if doc_type == "csv":
                self.knowledge_bases[doc_type] = CSVKnowledgeBase(filepaths=valid_paths)
            elif doc_type == "pdf":
                self.knowledge_bases[doc_type] = PDFKnowledgeBase(filepaths=valid_paths)
            elif doc_type == "text" or doc_type == "txt":
                self.knowledge_bases[doc_type] = TextKnowledgeBase(filepaths=valid_paths)
            elif doc_type == "json":
                self.knowledge_bases[doc_type] = JSONKnowledgeBase(filepaths=valid_paths)
            elif doc_type == "md":
                # Enhanced markdown handling - create specialized knowledge base
                self.knowledge_bases["markdown"] = TextKnowledgeBase(filepaths=valid_paths)
                # Pre-process markdown files for better retrieval
                self._preprocess_markdown_files(valid_paths)
            else:
                logger.warning(f"Unsupported document type: {doc_type}")
                return False
            
            logger.info(f"Initialized {doc_type} knowledge base with {len(valid_paths)} files")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {doc_type} knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _preprocess_markdown_files(self, markdown_paths: List[str]) -> None:
        """Pre-process markdown files to extract structured data for better retrieval.
        
        This builds a cache of markdown content with headings, sections, and keywords
        for faster and more accurate retrieval later.
        
        Args:
            markdown_paths: List of paths to markdown files
        """
        for md_path in markdown_paths:
            try:
                with open(md_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                file_name = os.path.basename(md_path)
                
                # Extract all headings with their levels and positions
                heading_pattern = r'^(#+)\s+(.+?)$'
                headings = list(re.finditer(heading_pattern, content, re.MULTILINE))
                
                # Build sections (heading + content until next same-level or higher heading)
                sections = []
                
                for i, match in enumerate(headings):
                    header_level = len(match.group(1))
                    header_text = match.group(2)
                    
                    # Determine section end
                    if i < len(headings) - 1:
                        # Find next heading of same or higher level
                        end_pos = None
                        for j in range(i + 1, len(headings)):
                            next_level = len(headings[j].group(1))
                            if next_level <= header_level:
                                end_pos = headings[j].start()
                                break
                        
                        if end_pos is None:
                            # No higher level heading found, go to end
                            section_content = content[match.end():].strip()
                        else:
                            section_content = content[match.end():end_pos].strip()
                    else:
                        # Last heading, take content until end
                        section_content = content[match.end():].strip()
                    
                    sections.append({
                        'level': header_level,
                        'heading': header_text,
                        'content': section_content,
                        'keywords': self._extract_keywords_from_text(header_text + " " + section_content)
                    })
                
                # Store structured content in cache
                self.markdown_cache[md_path] = {
                    'filename': file_name,
                    'headings': [{'level': len(h.group(1)), 'text': h.group(2)} for h in headings],
                    'sections': sections,
                    'full_content': content,
                    'keywords': self._extract_keywords_from_text(content)
                }
                
                logger.info(f"Preprocessed markdown file: {file_name} with {len(sections)} sections")
                
            except Exception as e:
                logger.error(f"Error preprocessing markdown file {md_path}: {str(e)}")
                
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract meaningful keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - remove stopwords and keep longer words
        # In a production system, this would use more sophisticated NLP
        stopwords = {'the', 'and', 'or', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 
                    'by', 'about', 'as', 'of', 'that', 'this', 'be', 'are', 'is', 'was'}
        
        # Extract words, lowercase them, filter out short words and stopwords
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stopwords]
        
        # Count word frequency
        word_counts = {}
        for word in keywords:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Get most frequent keywords (normalize by document length)
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [k for k, v in sorted_keywords[:25]]  # Return top 25 keywords
        
    def find_relevant_markdown_for_query(self, query: str, limit_attempts: bool = True) -> List[Dict[str, Any]]:
        """Find the most relevant markdown documents for a query.
        
        Args:
            query: The user's query
            limit_attempts: Whether to limit search attempts
            
        Returns:
            List of relevant markdown documents with details
        """
        # Check if we've exceeded the search attempt limit for this query
        if limit_attempts and query in self.search_attempts:
            if self.search_attempts[query] >= 2:
                logger.info(f"Reached maximum search attempts (2) for query: {query}")
                # Return any previous results
                if query in self.document_relevance_scores:
                    return self.document_relevance_scores[query]
                return []
            else:
                self.search_attempts[query] += 1
        else:
            self.search_attempts[query] = 1
        
        # Extract keywords from query
        query_keywords = self._extract_keywords_from_text(query)
        
        # Score each markdown document
        scored_documents = []
        
        for md_path, md_data in self.markdown_cache.items():
            # Calculate document relevance score
            score = 0
            
            # 1. Check for keyword matches in headings (high weight)
            for heading in md_data['headings']:
                heading_text = heading['text'].lower()
                for keyword in query_keywords:
                    if keyword in heading_text:
                        score += 5  # High weight for heading matches
            
            # 2. Check each section for relevance
            relevant_sections = []
            
            for section in md_data['sections']:
                section_score = 0
                
                # Check heading match
                heading_text = section['heading'].lower()
                for keyword in query_keywords:
                    if keyword in heading_text:
                        section_score += 3
                
                # Check content match
                content_text = section['content'].lower()
                for keyword in query_keywords:
                    keyword_count = content_text.count(keyword.lower())
                    if keyword_count > 0:
                        section_score += min(keyword_count, 5)  # Cap at 5 per keyword
                
                # If section is relevant, add it
                if section_score > 2:
                    relevant_sections.append({
                        'heading': section['heading'],
                        'content': section['content'],
                        'score': section_score
                    })
                    
                    # Add to total document score
                    score += section_score
            
            # Only include document if it has some relevance
            if score > 0:
                scored_documents.append({
                    'path': md_path,
                    'filename': md_data['filename'],
                    'score': score,
                    'relevant_sections': sorted(relevant_sections, key=lambda x: x['score'], reverse=True)[:3]  # Top 3 sections
                })
        
        # Sort by relevance score
        relevant_docs = sorted(scored_documents, key=lambda x: x['score'], reverse=True)
        
        # Store results for future reference
        self.document_relevance_scores[query] = relevant_docs
        
        return relevant_docs[:5]  # Return top 5 most relevant
    
    def create_combined_knowledge_base(self) -> bool:
        """Create a combined knowledge base from all initialized knowledge bases.
        
        Returns:
            Boolean indicating success
        """
        try:
            if len(self.knowledge_bases) < 1:
                logger.warning("No knowledge bases available to combine")
                return False
            
            # Create a combined knowledge base
            self.combined_knowledge = CombinedKnowledgeBase(
                knowledge_bases=list(self.knowledge_bases.values())
            )
            
            logger.info(f"Created combined knowledge base with {len(self.knowledge_bases)} sources")
            return True
            
        except Exception as e:
            logger.error(f"Error creating combined knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def search_knowledge(self, query: str, knowledge_types: Optional[List[str]] = None, 
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base for information relevant to a query.
        
        Args:
            query: The search query
            knowledge_types: Optional list of knowledge base types to search
            top_k: Number of top results to return
            
        Returns:
            List of search results with sources and content
        """
        try:
            if not self.combined_knowledge and not self.knowledge_bases:
                return [{"error": "No knowledge bases available for search"}]
            
            results = []
            
            # If specific knowledge types are requested
            if knowledge_types:
                for k_type in knowledge_types:
                    if k_type in self.knowledge_bases:
                        kb_results = self.knowledge_bases[k_type].search(query=query, top_k=top_k)
                        
                        # Format results
                        for result in kb_results:
                            results.append({
                                "source": result.metadata.get("source", "Unknown"),
                                "content": result.content,
                                "type": k_type,
                                "metadata": result.metadata
                            })
            
            # Otherwise search the combined knowledge base
            elif self.combined_knowledge:
                combined_results = self.combined_knowledge.search(query=query, top_k=top_k)
                
                # Format results
                for result in combined_results:
                    results.append({
                        "source": result.metadata.get("source", "Unknown"),
                        "content": result.content,
                        "type": result.metadata.get("type", "Unknown"),
                        "metadata": result.metadata
                    })
            
            # If no combined knowledge but we have individual knowledge bases
            elif self.knowledge_bases:
                for k_type, kb in self.knowledge_bases.items():
                    kb_results = kb.search(query=query, top_k=min(3, top_k))
                    
                    # Format results
                    for result in kb_results:
                        results.append({
                            "source": result.metadata.get("source", "Unknown"),
                            "content": result.content,
                            "type": k_type,
                            "metadata": result.metadata
                        })
            
            # Sort by relevance if we have mixed results
            # (This is simplified - in a real system you'd want a better ranking method)
            if len(results) > top_k:
                results = results[:top_k]
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return [{"error": f"Search error: {str(e)}"}]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text for knowledge graph.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        try:
            # Check if we already extracted entities from this text
            text_hash = hash(text[:1000])  # Use first 1000 chars as approximation
            if text_hash in self.entity_cache:
                return self.entity_cache[text_hash]
            
            # In a real implementation, this would use a proper NER system
            # Here's a simplified version using regex patterns
            
            entities = []
            
            # Extract potential named entities (capitalized words)
            named_entity_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
            named_entities = re.findall(named_entity_pattern, text)
            
            # Filter out common words that start sentences
            common_words = {"The", "A", "An", "This", "That", "These", "Those", "It", "They", "She", "He", "I", "We", "You"}
            named_entities = [e for e in named_entities if e not in common_words]
            
            # Add to entities list with type "name"
            for entity in named_entities:
                entities.append({
                    "text": entity,
                    "type": "name",
                    "start": text.find(entity),
                    "end": text.find(entity) + len(entity)
                })
            
            # Extract dates
            date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b'
            dates = re.findall(date_pattern, text)
            
            # Add to entities list with type "date"
            for date in dates:
                entities.append({
                    "text": date,
                    "type": "date",
                    "start": text.find(date),
                    "end": text.find(date) + len(date)
                })
            
            # Extract numbers
            number_pattern = r'\b\d+(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion))?\b'
            numbers = re.findall(number_pattern, text)
            
            # Add to entities list with type "number"
            for number in numbers:
                entities.append({
                    "text": number,
                    "type": "number",
                    "start": text.find(number),
                    "end": text.find(number) + len(number)
                })
            
            # Extract URLs
            url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
            urls = re.findall(url_pattern, text)
            
            # Add to entities list with type "url"
            for url in urls:
                entities.append({
                    "text": url,
                    "type": "url",
                    "start": text.find(url),
                    "end": text.find(url) + len(url)
                })
            
            # Cache the results
            self.entity_cache[text_hash] = entities
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities.
        
        Args:
            text: Text to extract relationships from
            entities: List of entities
            
        Returns:
            List of entity relationships
        """
        try:
            # We'll make a simplified relationship extractor
            # In a real system, you'd use a language model or dependency parser
            
            relationships = []
            
            # If we have fewer than 2 entities, no relationships to extract
            if len(entities) < 2:
                return relationships
            
            # For each pair of entities, try to find a connecting verb or preposition
            for i, entity1 in enumerate(entities):
                for entity2 in enumerate(entities[i+1:], i+1):
                    idx2, entity2 = entity2
                    
                    # Get the text between the two entities
                    if entity1["end"] < entity2["start"]:
                        between_text = text[entity1["end"]:entity2["start"]]
                    else:
                        between_text = text[entity2["end"]:entity1["start"]]
                    
                    # Look for relationship indicators
                    if re.search(r'\b(?:is|are|was|were|has|have|had|include[ds]?|contain[ds]?|consist[ds]? of)\b', between_text):
                        relationships.append({
                            "from": i,
                            "to": idx2,
                            "type": "has_property"
                        })
                    elif re.search(r'\b(?:cause[ds]?|create[ds]?|make[s]?|generate[ds]?|produce[ds]?)\b', between_text):
                        relationships.append({
                            "from": i,
                            "to": idx2,
                            "type": "causes"
                        })
                    elif re.search(r'\b(?:use[ds]?|utilize[ds]?|employ[ds]?|require[ds]?|need[ds]?)\b', between_text):
                        relationships.append({
                            "from": i,
                            "to": idx2,
                            "type": "uses"
                        })
                    elif re.search(r'\b(?:part of|belongs to|in|within|inside)\b', between_text):
                        relationships.append({
                            "from": i,
                            "to": idx2,
                            "type": "part_of"
                        })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def update_knowledge_graph(self, text: str, source: str) -> bool:
        """Update the knowledge graph with information from text.
        
        Args:
            text: Text to extract knowledge from
            source: Source of the text
            
        Returns:
            Boolean indicating success
        """
        try:
            # Extract entities
            entities = self.extract_entities(text)
            
            # Extract relationships
            relationships = self.extract_relationships(text, entities)
            
            # Add entities to knowledge graph
            for i, entity in enumerate(entities):
                entity_id = f"{source}_{i}"
                self.knowledge_graph.add_node(entity_id, **entity, source=source)
            
            # Add relationships to knowledge graph
            for rel in relationships:
                from_id = f"{source}_{rel['from']}"
                to_id = f"{source}_{rel['to']}"
                self.knowledge_graph.add_edge(from_id, to_id, type=rel["type"])
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def query_knowledge_graph(self, query: str, entity_type: Optional[str] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge graph for entities or relationships.
        
        Args:
            query: Query string to match against entities
            entity_type: Optional filter for entity type
            limit: Maximum number of results to return
            
        Returns:
            List of matching entities with their relationships
        """
        try:
            if not self.knowledge_graph.nodes:
                return [{"error": "Knowledge graph is empty"}]
            
            results = []
            matched_nodes = []
            
            # Find nodes that match the query
            for node, attrs in self.knowledge_graph.nodes(data=True):
                # Check if node text contains the query
                if 'text' in attrs and query.lower() in attrs['text'].lower():
                    # Check entity type if specified
                    if entity_type and attrs.get('type') != entity_type:
                        continue
                    
                    matched_nodes.append(node)
            
            # Get results for each matched node
            for node in matched_nodes[:limit]:
                # Get node attributes
                attrs = self.knowledge_graph.nodes[node]
                
                # Get related nodes
                related = []
                
                # Outgoing relationships (node -> other)
                for _, target, edge_attrs in self.knowledge_graph.out_edges(node, data=True):
                    target_attrs = self.knowledge_graph.nodes[target]
                    related.append({
                        "text": target_attrs.get('text', ''),
                        "type": target_attrs.get('type', ''),
                        "relationship": edge_attrs.get('type', 'related_to'),
                        "direction": "outgoing"
                    })
                
                # Incoming relationships (other -> node)
                for source, _, edge_attrs in self.knowledge_graph.in_edges(node, data=True):
                    source_attrs = self.knowledge_graph.nodes[source]
                    related.append({
                        "text": source_attrs.get('text', ''),
                        "type": source_attrs.get('type', ''),
                        "relationship": edge_attrs.get('type', 'related_to'),
                        "direction": "incoming"
                    })
                
                # Add to results
                results.append({
                    "entity": attrs.get('text', ''),
                    "type": attrs.get('type', ''),
                    "source": attrs.get('source', ''),
                    "related_entities": related
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {str(e)}")
            logger.error(traceback.format_exc())
            return [{"error": f"Query error: {str(e)}"}]
    
    def clear_knowledge_graph(self) -> bool:
        """Clear the knowledge graph.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.knowledge_graph.clear()
            self.entity_cache.clear()
            logger.info("Knowledge graph cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing knowledge graph: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            if not self.knowledge_graph.nodes:
                return {"entities": 0, "relationships": 0, "sources": 0}
            
            # Count entities by type
            entity_types = {}
            for _, attrs in self.knowledge_graph.nodes(data=True):
                entity_type = attrs.get('type', 'unknown')
                if entity_type in entity_types:
                    entity_types[entity_type] += 1
                else:
                    entity_types[entity_type] = 1
            
            # Count relationships by type
            relationship_types = {}
            for _, _, attrs in self.knowledge_graph.edges(data=True):
                rel_type = attrs.get('type', 'unknown')
                if rel_type in relationship_types:
                    relationship_types[rel_type] += 1
                else:
                    relationship_types[rel_type] = 1
            
            # Count unique sources
            sources = set()
            for _, attrs in self.knowledge_graph.nodes(data=True):
                if 'source' in attrs:
                    sources.add(attrs['source'])
            
            return {
                "entities": self.knowledge_graph.number_of_nodes(),
                "relationships": self.knowledge_graph.number_of_edges(),
                "sources": len(sources),
                "entity_types": entity_types,
                "relationship_types": relationship_types
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge graph stats: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
            
    def convert_kb_to_dict(self, knowledge_base) -> Dict[str, Any]:
        """Convert a knowledge base to a dictionary representation.
        
        Args:
            knowledge_base: The knowledge base to convert
            
        Returns:
            Dictionary representation of the knowledge base
        """
        try:
            kb_dict = {
                "type": type(knowledge_base).__name__,
                "documents": []
            }
            
            # Try to get documents from the knowledge base
            try:
                documents = knowledge_base.knowledge
                for i, doc in enumerate(documents[:10]):  # Limit to first 10 docs
                    kb_dict["documents"].append({
                        "index": i,
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "metadata": doc.metadata
                    })
                
                if len(documents) > 10:
                    kb_dict["total_documents"] = len(documents)
                    kb_dict["showing"] = "first 10 documents"
            except Exception as inner_e:
                kb_dict["documents"] = [{"error": f"Could not retrieve documents: {str(inner_e)}"}]
            
            return kb_dict
            
        except Exception as e:
            logger.error(f"Error converting knowledge base to dict: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)} 