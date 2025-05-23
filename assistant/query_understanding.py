import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import ollama
try:
    from .Logger import LoggerSetup
except ImportError:
    from Logger import LoggerSetup

class QueryUnderstanding:
    """
    Class for analyzing and understanding user queries, including intent detection,
    entity extraction, and keyword identification.
    """
    
    def __init__(self, default_model=None):
        """
        Initialize the query understanding system
        
        Args:
            default_model: Default LLM model to use for zero-shot classification
            logger: Optional logger instance
        """
        self.logger = LoggerSetup(rewrite=False, verbose=True)
        self.default_model = default_model or "Llama3.2-Vision:latest"
        
        # Define possible intents for zero-shot classification
        self.intent_classes = [
            "dataset_inquiry",     # User asking about data or datasets
            "file_operation",      # User wanting to open, view, or save files
            "spatial_analysis",    # Requests related to geographic or spatial analysis
            "temporal_analysis",   # Requests about time series or temporal trends
            "comparison",          # Comparing two or more entities or datasets
            "trend_analysis",      # Analyzing trends or patterns
            "help_request",        # User asking for help or instructions
            "conversation",        # General conversational queries (greetings, etc.)
            "general_inquiry",     # Any other general questions
            "analyze"              # Specific requests to analyze something
        ]
        
        # Keywords to look for in queries
        self.known_keywords = [
            "precipitation", "temperature", "climate", "weather", 
            "rainfall", "elevation", "population", "vegetation",
            "landcover", "urban", "rural", "agriculture", "forest",
            "water", "river", "lake", "ocean", "mountain", "coastal",
            "inland", "drought", "flood", "hurricane", "earthquake",
            "disaster", "emergency", "response", "planning", "conservation",
            "analysis", "visualization", "statistics", "model", "prediction",
            "forecast", "historic", "current", "future", "change", "trend"
        ]
        
        # Load known dataset vocabulary if available
        self.known_datasets = [
            "precipitation", "temperature", "climate", "landcover",
            "elevation", "population", "census", "demographics", 
            "satellite", "imagery", "remote_sensing", "NDVI",
            "MODIS", "Landsat", "SRTM", "LiDAR", "DEM", "DTM",
            "hydrology", "watershed", "runoff", "soil_moisture"
        ]
        
        self.logger.info("QueryUnderstanding initialized")
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze a user query to extract intent, entities, keywords, and other information
        
        Args:
            query: The user's query string
            
        Returns:
            dict: Analysis of the query with extracted information
        """
        self.logger.debug(f"Analyzing query: {query}")
        
        # Initialize analysis dictionary
        analysis = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "intent": None,
            "entities": [],
            "keywords": [],
            "dataset_references": []
        }
        
        # Detect intent
        intent, confidence = self._detect_intent(query)
        analysis["intent"] = intent
        analysis["intent_confidence"] = confidence
        
        # Extract entities
        entities = self._extract_entities(query, intent)
        analysis["entities"] = entities
        
        # Extract keywords
        keywords = self._extract_keywords(query)
        analysis["keywords"] = keywords
        
        # Look for dataset references
        dataset_references = self._extract_dataset_references(query)
        analysis["dataset_references"] = dataset_references
        
        # Determine if query is spatial
        spatial_indicators = ["map", "location", "geographic", "spatial", "region", "area", "where"]
        analysis["is_spatial"] = any(indicator in query.lower() for indicator in spatial_indicators)
        
        # Determine if query is temporal
        temporal_indicators = ["time", "date", "period", "year", "month", "when", "history", "trend", "temporal"]
        analysis["is_temporal"] = any(indicator in query.lower() for indicator in temporal_indicators)
        
        self.logger.debug(f"Query analysis: {json.dumps(analysis, indent=2)}")
        return analysis
    
    def enhance_query(self, query: str, query_info: Dict, memory_system) -> Dict:
        """
        Enhance the query with additional context from memory system
        
        Args:
            query: The original query string
            query_info: The analysis of the query
            memory_system: Memory system for retrieving context
            
        Returns:
            dict: Enhanced query information
        """
        self.logger.debug(f"Enhancing query: {query}")
        
        # Start with a copy of the query info
        enhanced_query = query_info.copy()
        
        # Get conversation history for context
        conversation_history = memory_system.get_conversation_history(limit=3)
        
        # Add conversation history
        enhanced_query["conversation_history"] = conversation_history
        
        # Add recently mentioned datasets
        mentioned_datasets = memory_system.get_mentioned_datasets()
        if mentioned_datasets:
            enhanced_query["recently_mentioned_datasets"] = mentioned_datasets
            
        # Add recently mentioned files
        mentioned_files = memory_system.get_mentioned_files()
        if mentioned_files:
            enhanced_query["recently_mentioned_files"] = mentioned_files
        
        # Look for additional information based on intent
        intent = query_info.get("intent")
        
        # For dataset inquiries, add related datasets
        if intent == "dataset_inquiry" or "dataset_references" in query_info:
            datasets = query_info.get("dataset_references", [])
            
            # If we have a dataset, add information about it
            for dataset in datasets:
                dataset_info = memory_system.get_dataset_info(dataset)
                if dataset_info:
                    if "dataset_info" not in enhanced_query:
                        enhanced_query["dataset_info"] = {}
                    enhanced_query["dataset_info"][dataset] = dataset_info
        
        # For file operations, try to resolve file references
        if intent == "file_operation" and "entities" in query_info:
            file_entities = [entity for entity in query_info["entities"] 
                           if entity.get("type") == "file"]
            
            if file_entities:
                resolved_files = []
                for entity in file_entities:
                    file_name = entity.get("value", "")
                    files = memory_system.get_related_files(file_name, 
                                                          query_info.get("keywords", []))
                    resolved_files.extend(files)
                
                if resolved_files:
                    enhanced_query["resolved_files"] = resolved_files
        
        # For spatial queries, add geospatial context
        if query_info.get("is_spatial"):
            # Get any location entities
            location_entities = [entity["value"] for entity in query_info.get("entities", [])
                               if entity.get("type") == "location"]
            
            if location_entities:
                enhanced_query["geospatial_context"] = {
                    "locations": location_entities
                }
        
        # For temporal queries, add temporal context
        if query_info.get("is_temporal"):
            # Get any time entities
            time_entities = [entity["value"] for entity in query_info.get("entities", [])
                           if entity.get("type") == "time"]
            
            if time_entities:
                enhanced_query["temporal_context"] = {
                    "time_periods": time_entities
                }
        
        # Add enhanced keywords
        if "keywords" in query_info:
            # Start with original keywords
            enhanced_keywords = query_info["keywords"].copy()
            
            # Add keywords from conversation history
            for interaction in conversation_history:
                if "query_analysis" in interaction and "keywords" in interaction["query_analysis"]:
                    hist_keywords = interaction["query_analysis"]["keywords"]
                    # Add relevant keywords that aren't already in the list
                    for keyword in hist_keywords:
                        if keyword not in enhanced_keywords:
                            enhanced_keywords.append(keyword)
            
            # Prioritize and filter keywords (top 10)
            if len(enhanced_keywords) > 10:
                # Keep original keywords and fill with history keywords
                orig_count = len(query_info["keywords"])
                enhanced_keywords = query_info["keywords"] + enhanced_keywords[orig_count:10]
            
            enhanced_query["enhanced_keywords"] = enhanced_keywords
        
        self.logger.debug(f"Enhanced query with additional context")
        return enhanced_query
    
    def _detect_intent(self, query: str, model=None) -> tuple:
        """
        Detect the intent of a query using zero-shot classification with LLM
        
        Args:
            query: The user's query string
            model: Optional model name to override the default
            
        Returns:
            tuple: (intent name, confidence score)
        """
        # Special case for common file listing commands for backward compatibility
        list_patterns = [
            r"(?i)list\s+(?:all\s+)?(\w+)(?:\s+files)?",
            r"(?i)show\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(\w+)(?:\s+files)?",
            r"(?i)what\s+(\w+)(?:\s+files)?\s+(?:do\s+you\s+have|are\s+available)"
        ]
        
        for pattern in list_patterns:
            if re.search(pattern, query):
                return "file_operation", 0.9
        
        # Special case for analyze commands for backward compatibility
        analyze_patterns = [r"(?i)anal[yz]e\s+([\w\.\-]+)"]
        for pattern in analyze_patterns:
            if re.search(pattern, query):
                return "analyze", 0.9
                
        # Use zero-shot classification with LLM for all other queries
        try:
            # Use provided model or fall back to default
            llm_model = model or self.default_model
            
            # Prepare a prompt for zero-shot classification
            prompt = f"""Classify the following query into one of these categories:
            {', '.join(self.intent_classes)}
            
            Query: "{query}"
            
            Respond with only the intent class name and a confidence score between 0 and 1, 
            in the format: "intent_class|confidence".
            For example: "dataset_inquiry|0.85"
            """
            
            self.logger.debug(f"Sending zero-shot classification prompt to LLM using model {llm_model}")
            
            # Get classification from LLM
            response = ollama.generate(
                model=llm_model,
                prompt=prompt,
            )
            
            # Parse response
            prediction = response.response.strip()
            self.logger.debug(f"LLM classification result: {prediction}")
            
            # Extract intent and confidence
            if "|" in prediction:
                intent, confidence_str = prediction.split("|", 1)
                intent = intent.strip()
                
                # Make sure intent is in valid classes
                if intent not in self.intent_classes:
                    self.logger.warning(f"LLM returned invalid intent: {intent}, defaulting to general_inquiry")
                    intent = "general_inquiry"
                
                try:
                    confidence = float(confidence_str)
                    if confidence < 0 or confidence > 1:
                        confidence = 0.7  # Default if out of range
                except ValueError:
                    confidence = 0.7  # Default if confidence is not a valid float
            else:
                # Fallback if response format is incorrect
                self.logger.warning(f"LLM response format incorrect: {prediction}")
                intent = prediction if prediction in self.intent_classes else "general_inquiry"
                confidence = 0.6
            
            return intent, confidence
            
        except Exception as e:
            # If LLM classification fails, fall back to a safe default
            self.logger.error(f"Error in zero-shot classification: {str(e)}")
            return "general_inquiry", 0.5
    
    def _extract_entities(self, query: str, intent: str) -> List[Dict]:
        """
        Extract entities from the query based on intent
        
        Args:
            query: The user's query string
            intent: Detected intent
            
        Returns:
            list: List of entity dictionaries
        """
        entities = []
        
        # Extract file names
        if intent == "file_operation":
            file_patterns = [
                r"(?i)file\s+([a-zA-Z0-9\._\-]+)",
                r"(?i)(?:open|read|show|display|save|write|create)\s+([a-zA-Z0-9\._\-]+\.[a-zA-Z0-9]+)",
            ]
            
            for pattern in file_patterns:
                matches = re.finditer(pattern, query)
                for match in matches:
                    file_name = match.group(1).strip()
                    entities.append({
                        "type": "file",
                        "value": file_name,
                        "start": match.start(1),
                        "end": match.end(1)
                    })
        
        # Extract locations
        location_patterns = [
            r"(?i)(?:in|at|near|around)\s+([A-Z][a-zA-Z\s,]+)(?:\.|\?|$|\s)",
            r"(?i)(?:of|for)\s+([A-Z][a-zA-Z\s,]+)(?:\.|\?|$|\s)"
        ]
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                location = match.group(1).strip()
                if location and not any(entity.get("value") == location for entity in entities):
                    entities.append({
                        "type": "location",
                        "value": location,
                        "start": match.start(1),
                        "end": match.end(1)
                    })
        
        # Extract time periods
        time_patterns = [
            r"(?i)(?:in|from|during|for)\s+(\d{4}(?:\s*-\s*\d{4})?)",
            r"(?i)(?:in|from|during|for)\s+([A-Z][a-z]+\s+\d{4})",
            r"(?i)(?:last|past|previous)\s+(\d+\s+(?:year|month|day|week)s?)"
        ]
        
        for pattern in time_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                time_period = match.group(1).strip()
                entities.append({
                    "type": "time",
                    "value": time_period,
                    "start": match.start(1),
                    "end": match.end(1)
                })
        
        # Extract quantities
        quantity_patterns = [
            r"(\d+(?:\.\d+)?)\s+(mm|cm|m|km|inch|feet|acre|hectare|percent|%)"
        ]
        
        for pattern in quantity_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                value = match.group(1)
                unit = match.group(2)
                entities.append({
                    "type": "quantity",
                    "value": float(value),
                    "unit": unit,
                    "full_text": f"{value} {unit}",
                    "start": match.start(),
                    "end": match.end()
                })
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from the query
        
        Args:
            query: The user's query string
            
        Returns:
            list: List of keywords
        """
        # Convert to lowercase and split into words
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Filter stopwords and extract keywords
        stopwords = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "in", "on", "at", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below", "to",
            "from", "up", "down", "of", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how", "all",
            "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
            "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
            "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
            "wasn", "weren", "won", "wouldn", "show", "tell", "want", "like", "need",
            "help", "do", "does", "did", "have", "has", "had", "would", "could", "should",
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those"
        ]
        
        # Extract keywords that are in our known list or not in stopwords
        keywords = []
        for word in words:
            if len(word) > 2 and word not in stopwords:
                if word in self.known_keywords:
                    keywords.append(word)
                elif word not in keywords:
                    keywords.append(word)
        
        # Also check for multi-word keywords
        for keyword in self.known_keywords:
            if " " in keyword and keyword.lower() in query_lower:
                keywords.append(keyword)
        
        return keywords
    
    def _extract_dataset_references(self, query: str) -> List[str]:
        """
        Extract references to known datasets
        
        Args:
            query: The user's query string
            
        Returns:
            list: List of dataset references
        """
        query_lower = query.lower()
        datasets = []
        
        # Check for known datasets
        for dataset in self.known_datasets:
            if dataset.lower() in query_lower:
                datasets.append(dataset)
        
        # Check for common dataset patterns
        dataset_patterns = [
            r"(?i)(?:dataset|data\s+set|data)\s+(?:called|named)\s+([a-zA-Z0-9_\-\s]+)",
            r"(?i)([a-zA-Z0-9_\-]+)\s+(?:dataset|data\s+set|data)"
        ]
        
        for pattern in dataset_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                dataset_name = match.group(1).strip()
                if dataset_name and dataset_name.lower() not in [d.lower() for d in datasets]:
                    datasets.append(dataset_name)
        
        return datasets 