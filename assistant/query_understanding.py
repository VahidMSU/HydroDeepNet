import re
from typing import Dict, List, Any, Optional

class QueryUnderstanding:
    """
    Component responsible for understanding and enhancing user queries
    by extracting intent, keywords, and other relevant information.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the QueryUnderstanding component
        """
        self.logger = logger
        self.logger.info("Initializing Query Understanding component")
        
        # Define intents and patterns
        self.intent_patterns = {
            "search": [
                r"(?i)find|search|look for|locate|get|retrieve",
                r"(?i)where (is|are)|what files",
                r"(?i)any (information|data|files) (on|about)"
            ],
            "analyze": [
                r"(?i)analyze|analyse|study|examine|investigate|show|display|visualize|plot",
                r"(?i)what (is|are) (in|inside|contained|the data|the content)",
                r"(?i)what (is|does) .+ (show|contain|tell)",
                r"(?i)compare|contrast|correlation|interpret|explain",
                r"(?i)data (in|from|of)|file (content|data)"
            ],
            "help": [
                r"(?i)how (can|do) (i|you)|help me",
                r"(?i)what can you do|capabilities|features",
                r"(?i)explain how to"
            ],
            "geographic": [
                r"(?i)where (is|are)",
                r"(?i)location of|coordinates for",
                r"(?i)(city|country|state|region|area|place) of",
                r"(?i)near|nearby|close to",
                r"(?i)geographical|geospatial"
            ],
            "dataset_inquiry": [
                r"(?i)(what|tell me) about (the |)(\w+) data(set|)",
                r"(?i)summarize the (\w+) data",
                r"(?i)explain the (\w+) dataset",
                r"(?i)information (about|on|from) (\w+)",
                r"(?i)insights (from|about) (\w+)"
            ]
        }
        
        # Geographic entity patterns
        self.geo_patterns = [
            r"(?i)in ([A-Z][a-z]+([ \-'][A-Z][a-z]+)*)",  # Places with capitalized names
            r"(?i)near ([A-Z][a-z]+([ \-'][A-Z][a-z]+)*)",
            r"(?i)at ([A-Z][a-z]+([ \-'][A-Z][a-z]+)*)",
            r"(?i)from ([A-Z][a-z]+([ \-'][A-Z][a-z]+)*)",
            r"(?i)(city|town|village|country|state|province|region) of ([A-Z][a-z]+([ \-'][A-Z][a-z]+)*)"
        ]
        
        # File reference patterns
        self.file_patterns = [
            r"(?i)(\w+\.(csv|png|jpg|jpeg|md|txt))",  # Filename with extension
            r"(?i)(analyze|analyse|show|display|visualize|examine|open|check|view|see) (.+\.(csv|png|jpg|jpeg|md|txt))",
            r"(?i)(analyze|analyse|show|display|visualize|examine|open|check|view|see) (the |a |this |that )?(.+) (file|image|csv|document|data|visualization|plot|chart|map)"
        ]
        
        # Known datasets and their related keywords
        self.known_datasets = {
            "cdl": ["cdl", "cropland", "crop", "agriculture", "land use", "land cover", "crop data layer", "usda", "crop type"],
            "climate_change": ["climate change", "climate", "global warming", "temperature change", "precipitation change", "future climate", "climate projections", "climate scenarios"],
            "gov_units": ["gov units", "governmental units", "administrative boundaries", "counties", "states", "boundaries", "administrative regions", "political boundaries"],
            "groundwater": ["groundwater", "aquifer", "well", "water table", "ground water", "subsurface water", "water level", "water depth", "hydrology"],
            "gssurgo": ["gssurgo", "soil", "sand", "clay", "silt", "organic matter", "bulk density", "soil properties", "soil survey", "usda soil"],
            "modis": ["modis", "evi", "ndvi", "lai", "vegetation", "mod13q1", "mod16a2", "mod15a2h", "satellite", "remote sensing", "vegetation index", "leaf area", "evapotranspiration", "et"],
            "nsrdb": ["nsrdb", "solar", "radiation", "solar power", "renewable energy", "irradiance", "pv", "photovoltaic", "solar potential", "insolation", "solar resource"],
            "prism": ["prism", "climate", "temperature", "precipitation", "rainfall", "weather", "drought", "historical climate", "climate data", "meteorology"],
            "snowdas": ["snow", "snowdas", "swe", "snow water equivalent", "snow cover", "snow depth", "snowpack", "snow accumulation", "snow melt"]
        }
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query to extract intent, keywords, and entities
        
        Args:
            query (str): The user's query
            
        Returns:
            dict: Analysis results including intent, keywords, and entities
        """
        self.logger.debug(f"Analyzing query: {query}")
        
        # Initialize query info
        query_info = {
            "original_query": query,
            "keywords": self._extract_keywords(query),
            "length": len(query),
            "intent": self._determine_intent(query),
            "entities": self._extract_entities(query)
        }
        
        # Extract file references if present
        file_references = self._extract_file_references(query)
        if file_references:
            query_info["file_references"] = file_references
            # If file references are found, consider it an analysis intent
            if query_info["intent"] == "search":
                query_info["intent"] = "analyze"
                self.logger.debug(f"Adjusted intent to 'analyze' due to file references")
        
        # Check for dataset-specific queries
        target_dataset = self._identify_dataset(query)
        if target_dataset:
            query_info["target_dataset"] = target_dataset
            query_info["dataset_terms"] = self.known_datasets[target_dataset]
            # If we're asking about a dataset and intent is search, adjust to dataset_inquiry
            if query_info["intent"] == "search":
                query_info["intent"] = "dataset_inquiry"
                self.logger.debug(f"Adjusted intent to 'dataset_inquiry' based on detected dataset: {target_dataset}")
        
        # Extract geographic entities if detected
        if query_info["intent"] == "geographic" or self._has_geographic_indicators(query):
            query_info["geographic_entities"] = self._extract_geographic_entities(query)
            
        self.logger.debug(f"Query analysis result: {query_info}")
        return query_info
    
    def _identify_dataset(self, query: str) -> Optional[str]:
        """
        Identify if the query is targeting a specific dataset
        
        Args:
            query (str): The query to analyze
            
        Returns:
            str or None: Dataset name if identified
        """
        query_lower = query.lower()
        
        # Handle common misspellings and alternative forms
        spelling_corrections = {
            "prims": "prism",
            "modiss": "modis",
            "nsrd": "nsrdb",
            "snowdass": "snowdas",
            "gssurgo soil": "gssurgo",
            "crop data": "cdl",
            "cropland data layer": "cdl",
            "climate change": "climate_change"
        }
        
        # Apply spelling corrections
        corrected_query = query_lower
        for misspelled, correct in spelling_corrections.items():
            if misspelled in query_lower:
                corrected_query = corrected_query.replace(misspelled, correct)
                self.logger.info(f"Corrected dataset spelling from '{misspelled}' to '{correct}'")
        
        # Check for direct mentions of dataset names first (highest priority)
        for dataset in self.known_datasets:
            # Exact dataset name match
            if dataset in corrected_query.split() or f"{dataset} data" in corrected_query:
                self.logger.info(f"Direct dataset match: {dataset}")
                return dataset
        
        # Check for partial dataset name matches (medium priority)
        partial_matches = []
        for dataset in self.known_datasets:
            if dataset in corrected_query:
                # Calculate match score based on how much of the dataset name is present
                match_score = len(dataset) / len(corrected_query)
                partial_matches.append((dataset, match_score))
        
        if partial_matches:
            # Sort by match score (descending)
            partial_matches.sort(key=lambda x: x[1], reverse=True)
            dataset = partial_matches[0][0]
            self.logger.info(f"Partial dataset name match: {dataset} (score: {partial_matches[0][1]:.2f})")
            return dataset
        
        # Check for keyword matches only if no direct or partial dataset name matches
        keyword_matches = {}
        for dataset, keywords in self.known_datasets.items():
            match_count = 0
            matched_terms = []
            
            for keyword in keywords:
                if keyword in corrected_query:
                    match_count += 1
                    matched_terms.append(keyword)
                    
                    # For exact matches of specific technical terms, give higher weight
                    if keyword in ["evi", "ndvi", "lai", "swe", "pv", "et"] and keyword in corrected_query.split():
                        match_count += 2  # Extra weight for exact technical term matches
            
            if match_count > 0:
                keyword_matches[dataset] = (match_count, matched_terms)
        
        if keyword_matches:
            # Find dataset with highest match count
            best_dataset = max(keyword_matches.items(), key=lambda x: x[1][0])
            dataset = best_dataset[0]
            match_count, matched_terms = best_dataset[1]
            
            # Only accept keyword matches if the match count is at least 2
            # or if one of the matched terms is a specific technical term
            specific_terms = ["evi", "ndvi", "lai", "swe", "pv", "et", "snowdas", "nsrdb", "gssurgo"]
            has_specific_term = any(term in specific_terms for term in matched_terms)
            
            if match_count >= 2 or has_specific_term:
                self.logger.info(f"Keyword match for dataset: {dataset} (score: {match_count}, terms: {matched_terms})")
                return dataset
            else:
                self.logger.warning(f"Weak keyword match rejected: {dataset} (score: {match_count}, terms: {matched_terms})")
                        
        return None
    
    def enhance_query(self, 
                     query: str, 
                     query_info: Dict[str, Any], 
                     memory_system) -> Dict[str, Any]:
        """
        Enhance the query using memory system information
        
        Args:
            query (str): Original query
            query_info (dict): Query analysis information
            memory_system: Memory system component
            
        Returns:
            dict: Enhanced query information
        """
        enhanced_query = query_info.copy()
        
        try:
            # Get related interactions
            related_interactions = memory_system.get_related_interactions(
                query, 
                query_info.get("keywords", [])
            )
            
            # Extract additional keywords from related interactions
            additional_keywords = set()
            for interaction in related_interactions[:3]:  # Use top 3 related interactions
                if "query_info" in interaction and "keywords" in interaction["query_info"]:
                    additional_keywords.update(interaction["query_info"]["keywords"])
            
            # Add new keywords (not already in the original set)
            new_keywords = additional_keywords - set(query_info["keywords"])
            if new_keywords:
                enhanced_query["enhanced_keywords"] = list(query_info["keywords"]) + list(new_keywords)
                self.logger.debug(f"Enhanced keywords: {enhanced_query['enhanced_keywords']}")
                
            # If we have a target dataset, add it to the enhanced keywords
            if "target_dataset" in query_info:
                dataset = query_info["target_dataset"]
                if "enhanced_keywords" not in enhanced_query:
                    enhanced_query["enhanced_keywords"] = list(query_info["keywords"])
                # Add dataset name and a few key terms
                enhanced_query["enhanced_keywords"].append(dataset)
                for term in self.known_datasets[dataset][:3]:  # Add top 3 related terms
                    if term not in enhanced_query["enhanced_keywords"]:
                        enhanced_query["enhanced_keywords"].append(term)
                
            # If we have file references, try to get full paths and add to enhanced query
            if "file_references" in query_info and query_info["file_references"]:
                file_refs = query_info["file_references"]
                file_paths = []
                
                # Look up each file reference in memory
                for file_ref in file_refs:
                    file_records = memory_system.get_related_files(file_ref, [file_ref.split(".")[0]], limit=3)
                    for record in file_records:
                        if "original_path" in record:
                            file_paths.append(record["original_path"])
                
                if file_paths:
                    enhanced_query["file_paths"] = file_paths
                    self.logger.debug(f"Enhanced with file paths: {enhanced_query['file_paths']}")
                
            # Enhance geographic understanding if applicable
            if "geographic_entities" in query_info and query_info["geographic_entities"]:
                geo_entities = query_info["geographic_entities"]
                
                # Get additional geographic info from memory
                enhanced_geo_info = {}
                for entity in geo_entities:
                    geo_info = memory_system.get_geographic_info(entity)
                    if geo_info:
                        enhanced_geo_info[entity] = geo_info
                
                if enhanced_geo_info:
                    enhanced_query["enhanced_geographic_info"] = enhanced_geo_info
            
            return enhanced_query
            
        except Exception as e:
            self.logger.error(f"Error enhancing query: {str(e)}", exc_info=True)
            return query_info  # Return original query_info if enhancement fails
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text
        
        Args:
            text (str): Text to extract keywords from
            
        Returns:
            list: List of keywords
        """
        # Remove common stop words
        stop_words = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
                     "in", "on", "at", "to", "for", "with", "by", "about", "like", 
                     "from", "of", "that", "this", "these", "those", "it", "they",
                     "file", "files", "show", "me", "please", "analyze", "analyse", "can", "you"}
        
        # Tokenize, clean, and filter
        words = re.findall(r'\b[a-zA-Z]\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Check for dataset keywords and prioritize them
        dataset_keywords = []
        for dataset, terms in self.known_datasets.items():
            if dataset in text.lower():
                dataset_keywords.append(dataset)
            for term in terms:
                if term in text.lower() and term not in dataset_keywords:
                    dataset_keywords.append(term)
        
        # Combine dataset keywords with regular keywords
        for dataset_keyword in dataset_keywords:
            if dataset_keyword not in keywords:
                keywords.insert(0, dataset_keyword)  # Insert at beginning to prioritize
        
        # Extract potential filenames (words with extensions)
        file_patterns = [r'\b\w+\.(csv|png|jpg|jpeg|md|txt)\b']
        for pattern in file_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                # Add the full filename to keywords if found
                full_match = text[text.lower().find(match[0])-len(match[0]):text.lower().find(match[0])+len(match[0])]
                if full_match not in keywords:
                    keywords.append(full_match)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = [k for k in keywords if not (k in seen or seen.add(k))]
        
        return unique_keywords
    
    def _determine_intent(self, query: str) -> str:
        """
        Determine the intent of a query
        
        Args:
            query (str): The query to analyze
            
        Returns:
            str: Detected intent
        """
        # Check if query is about a specific dataset
        if self._identify_dataset(query):
            # Look for dataset inquiry patterns
            for pattern in self.intent_patterns["dataset_inquiry"]:
                if re.search(pattern, query):
                    return "dataset_inquiry"
        
        # First check for file references which strongly indicate analysis intent
        file_references = self._extract_file_references(query)
        if file_references:
            return "analyze"
            
        # Check each intent pattern
        intent_scores = {"search": 0, "analyze": 0, "help": 0, "geographic": 0, "dataset_inquiry": 0}
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    intent_scores[intent] += 1
        
        # Get the intent with the highest score
        max_score = max(intent_scores.values())
        if max_score > 0:
            # If there's a tie, prioritize in this order: dataset_inquiry, analyze, search, geographic, help
            priority_order = ["dataset_inquiry", "analyze", "search", "geographic", "help"]
            for intent in priority_order:
                if intent_scores[intent] == max_score:
                    return intent
        
        # Default to search if no specific intent is detected
        return "search"
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract named entities from query
        
        Args:
            query (str): The query to analyze
            
        Returns:
            list: Detected entities
        """
        # Simple entity extraction based on capitalization
        # This is a placeholder for more sophisticated NER
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        return entities
    
    def _extract_file_references(self, query: str) -> List[str]:
        """
        Extract potential file references from the query
        
        Args:
            query (str): The query to analyze
            
        Returns:
            list: List of potential file references
        """
        file_references = []
        
        # Check for direct file mentions with extensions
        extensions = ["csv", "png", "jpg", "jpeg", "md", "txt"]
        ext_pattern = "|".join(extensions)
        direct_pattern = rf'\b\w+[-_\w]*\.({ext_pattern})\b'
        direct_matches = re.findall(direct_pattern, query, re.IGNORECASE)
        
        # Find the complete filename for each match
        for ext in direct_matches:
            # Find the position of this extension
            pos = query.lower().find(f".{ext.lower()}")
            if pos > 0:
                # Extract the filename
                start_pos = pos
                while start_pos > 0 and query[start_pos-1].isalnum() or query[start_pos-1] in "-_":
                    start_pos -= 1
                
                filename = query[start_pos:pos+len(ext)+1]
                if filename not in file_references:
                    file_references.append(filename)
        
        # Check for files mentioned after action verbs
        for pattern in self.file_patterns:
            matches = re.findall(pattern, query)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        # Different patterns will have different tuple structures
                        # Extract the filename or reference from the appropriate position
                        if len(match) == 2 and match[1] in extensions:
                            # Pattern 1: full filename with extension
                            filename = match[0]
                        elif len(match) >= 2:
                            # Pattern 2 or 3: verb + filename
                            filename = match[-2] if len(match) > 2 else match[-1]
                            if filename.lower() in extensions or filename.lower() in ["file", "image", "document"]:
                                continue  # Skip if we matched just "file" or "image" without a name
                        else:
                            continue  # Skip if we can't determine a filename
                            
                        if filename and filename not in file_references:
                            file_references.append(filename)
        
        return file_references
    
    def _has_geographic_indicators(self, query: str) -> bool:
        """
        Check if query has geographic indicators
        
        Args:
            query (str): The query to check
            
        Returns:
            bool: True if geographic indicators are present
        """
        geo_indicators = ["where", "location", "place", "area", "region", "city", 
                         "country", "state", "coordinates", "near", "nearby", 
                         "close to", "far from", "latitude", "longitude"]
        
        query_lower = query.lower()
        for indicator in geo_indicators:
            if indicator in query_lower:
                return True
        
        # Check for capitalized place names with geographic prepositions
        for pattern in self.geo_patterns:
            if re.search(pattern, query):
                return True
                
        return False
    
    def _extract_geographic_entities(self, query: str) -> List[str]:
        """
        Extract geographic entities from query
        
        Args:
            query (str): The query to analyze
            
        Returns:
            list: Detected geographic entities
        """
        entities = []
        
        # Extract from patterns
        for pattern in self.geo_patterns:
            matches = re.findall(pattern, query)
            if matches:
                # Handle different match formats based on capture groups
                for match in matches:
                    if isinstance(match, tuple):
                        # Get the place name from the tuple (second capture group)
                        entities.append(match[1])
                    else:
                        entities.append(match)
        
        # Backup method: look for capitalized words after geographic prepositions
        prepositions = ["in", "at", "near", "from", "to"]
        
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in prepositions and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word[0].isupper():
                    # Get the full entity (may span multiple capitalized words)
                    entity = next_word
                    j = i + 2
                    while j < len(words) and words[j][0].isupper():
                        entity += " " + words[j]
                        j += 1
                    
                    if entity not in entities:
                        entities.append(entity)
        
        # Remove duplicates and clean up
        unique_entities = []
        for entity in entities:
            cleaned = entity.strip(".,;:!?'\"")
            if cleaned and cleaned not in unique_entities:
                unique_entities.append(cleaned)
                
        return unique_entities 