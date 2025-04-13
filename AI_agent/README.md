# Enhanced Hydrology Report Analyzer

An advanced interface for analyzing hydrology reports, integrating:
- Specialized readers for different file types
- Persistent memory using SQLite
- Knowledge graph for concept relationships
- Advanced reasoning with conversational capabilities

## Features

- **Interactive Query Processing**: Natural language understanding of user requests
- **Multi-Modal Analysis**: Support for text, images, CSV data, and more
- **Contextual Memory**: Remembers previous analyses and user preferences
- **Knowledge Graph**: Builds connections between environmental concepts
- **Enhanced Reasoning**: Combines analysis from multiple sources into cohesive responses

## Setup

1. Install dependencies:
```bash
pip install -r requirements/requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

3. Set up environment variables:
```bash
# For OpenAI API access if needed
export OPENAI_API_KEY="your-api-key"

# For PostgreSQL vector database 
export PGVECTOR_CONNECTION="postgresql+psycopg://ai:ai@localhost:5432/ai"
```

## Usage

### Basic Usage

Run the interactive analyzer:
```bash
python integration.py --base-dir /path/to/reports
```

### Available Commands

- `help`: Shows available commands
- `list reports`: Lists available reports
- `list groups`: Lists groups in the current report
- `set report [report_id]`: Sets the current report
- `set group [group_id]`: Sets the current group
- `show knowledge`: Generates and displays the knowledge graph
- `show memory`: Shows insights remembered from previous analyses

### Natural Language Queries

You can also ask natural language questions:
- "Show me groundwater trends in this report"
- "Analyze climate change data and compare with soil moisture"
- "What do the crop rotation visualizations tell us?"
- "How do precipitation patterns affect groundwater levels?"

## Architecture

The system consists of several components:

- `interactive_agent.py`: Core interface for user interaction
- `ContextMemory.py`: Persistent memory storage using SQLite
- `KnowledgeGraph.py`: Concept relationship management
- `integration.py`: Integrates all components
- Specialized readers:
  - `text_reader.py`: Processes markdown and text files
  - `image_reader.py`: Analyzes images and visualizations
  - `csv_reader.py`: Processes tabular data
  - `json_reader.py`: Handles configuration files
  - `combine_reader.py`: Analyzes entire data groups

## Example

```
> Tell me about groundwater trends in the latest report

Switched to group: groundwater

Analysis shows declining water levels across the region, with an average drop of 0.8 meters over the past 5 years. The most significant decreases are observed in the southwestern quadrant where agricultural activity is highest.

Key findings:
- Recharge rates have decreased by 15% in correlation with reduced precipitation
- Pumping volumes increased 23% in agricultural areas
- Seasonal fluctuations show diminishing recovery during wet seasons

This suggests a potential sustainability issue if current extraction rates continue without mitigation measures.

You might also want to ask:
- How do these groundwater trends correlate with climate data?
- What are the projected impacts on agricultural productivity?
- Are there spatial patterns in the groundwater decline?
```

## Extending the System

To add new capabilities:
1. Create a new reader in `readers/` for additional file types
2. Update the file type mapping in `interactive_agent.py`
3. Add domain knowledge to `KnowledgeGraph.py` 

## Query Intent Recognition

The system features an intelligent query intent recognition system that can distinguish between different types of user queries:

1. **Information Queries**: Requests for information about available data files, groups, or content.
   - Example: "What data are in the CDL group?"
   - Response: Lists available files within the specified group with brief descriptions.

2. **Analysis Queries**: Requests for in-depth analysis of specific data.
   - Example: "Analyze the groundwater data"
   - Response: Performs a comprehensive analysis of the specified data.

3. **Conversational Queries**: Questions about concepts, relationships, or explanations.
   - Example: "Tell me more about soil moisture trends"
   - Response: Provides explanatory information drawing from previous analyses.

4. **Command Queries**: System commands for navigation and configuration.
   - Example: "list groups", "set report [ID]"
   - Response: Executes the specified command and provides feedback.

The intent recognition system uses natural language processing to identify the user's intention, allowing the agent to respond appropriately to different types of requests. This improves the user experience by providing just the right amount of information without unnecessary processing.

To test the query intent recognition system, run:

```bash
python test_query_intent.py
```

For an interactive demonstration:

```bash
python test_query_intent.py --interactive
``` 