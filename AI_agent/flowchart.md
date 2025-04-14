graph TD
    subgraph User_Interaction
        UserInput[User Prompt]
    end

    subgraph Core_Execution[run_ai.py / AIRunner]
        ParseArgs[Parse Arguments]
        CheckDeps[Check Dependencies]
        ModeDecision{Interactive or API?}

        UserInput --> ParseArgs
        ParseArgs --> CheckDeps
        CheckDeps --> ModeDecision
    end

    subgraph Interactive_Mode
        Integration[integration.py <br> EnhancedReportAnalyzer]
        InteractiveAgent[InteractiveReportAgent]
        ContextMem[Context Memory]
        KnowledgeGraph[Knowledge Graph]
        
        Integration -- Initializes --> InteractiveAgent
        Integration -- Initializes --> ContextMem
        Integration -- Initializes --> KnowledgeGraph
        Integration -- Processes Query --> InteractiveAgent
        InteractiveAgent -- Returns Response --> Integration
        Integration -- Updates --> KnowledgeGraph
        Integration -- Returns Response --> UserOutput[User Response]
    end

    subgraph API_Mode
        ListCommands{List Command?}
        ProcessQuery[Process Query]
        DirDiscover_API[dir_discover.py]
        OutputGenAPI[Generate JSON]
        
        ListCommands -- Yes --> DirDiscover_API
        DirDiscover_API -- Returns List --> OutputGenAPI
        ListCommands -- No --> ProcessQuery
        ProcessQuery -- Calls --> InteractiveAgent
        InteractiveAgent -- Returns --> OutputGenAPI
        OutputGenAPI --> UserOutput
    end

    subgraph Agent_Core[InteractiveReportAgent]
        AgentInit[Initialize Agent]
        DirDiscover_Agent[dir_discover.py]
        ProcessUserQuery[Process User Query]
        Ollama[Ollama (Intent Routing)]
        ActionRouter{Route Action}
        SelectFileReader{Select File Reader}
        
        AgentInit -- Loads Reports --> DirDiscover_Agent
        AgentInit -- Uses --> ContextMem
        ProcessUserQuery -- Handles Commands --> HistorySystem[History/System]
        ProcessUserQuery -- Gathers Context --> ContextMem
        ProcessUserQuery -- Calls LLM --> Ollama
        Ollama -- Returns Intent --> ActionRouter
        ActionRouter -- analyze_file --> SelectFileReader
        ActionRouter -- analyze_group --> CombineReader[combine_reader.py]
        ActionRouter -- list_* --> ListHandler[List Handler]
        ActionRouter -- clarify_* --> ClarificationHandler[Clarification]
        ActionRouter -- general --> GenericResponse[Generic Response]
        
        SelectFileReader -- .csv --> CSVReader[csv_reader.py]
        SelectFileReader -- .txt/.md --> TextReader[text_reader.py]
        SelectFileReader -- .png/.jpg --> ImageReader[image_reader.py]
        SelectFileReader -- .json --> JSONReader[json_reader.py]
        SelectFileReader -- website --> WebsiteReader[website_reader.py]
    end

    ModeDecision -- Interactive --> Integration
    ModeDecision -- API --> ListCommands