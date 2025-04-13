import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faRobot,
  faPaperPlane,
  faInfo,
  faQuestionCircle,
  faCog,
  faCheckCircle,
  faExclamationTriangle,
  faSpinner,
  faChevronDown,
  faChevronUp,
} from '@fortawesome/free-solid-svg-icons';

import {
  HydroGeoContainer,
  HydroGeoHeader,
  ChatContainer,
  ChatHeader,
  ChatMessagesContainer,
  MessageBubble,
  MessageList,
  ChatInputContainer,
  ThinkingIndicator,
  InputField,
  FormGroup,
} from '../../styles/HydroGeoDataset.tsx';

const HydroGeoAssistant = () => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gemini-1.5-flash');
  const [agnoStatus, setAgnoStatus] = useState({ connected: false, error: null });
  const [aboutExpanded, setAboutExpanded] = useState(false);
  const [samplesExpanded, setSamplesExpanded] = useState(false);
  const [settingsExpanded, setSettingsExpanded] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const initializedRef = useRef(false);

  // Models available for selection
  const availableModels = [
    { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash (Google)', provider: 'Google' },
    { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro (Google)', provider: 'Google', disabled: true },
    { id: 'claude-3-5-sonnet', name: 'Claude 3.5 Sonnet (Future)', provider: 'Anthropic', disabled: true },
  ];

  // Memoize the initializeAgent function to prevent infinite loops
  const initializeAgent = useCallback(async () => {
    try {
      setIsLoading(true);
      setAgnoStatus({ connected: false, error: null });

      // Don't reset chat history if we're just checking connection
      if (!sessionId) {
        setChatHistory([
          {
            type: 'bot',
            content:
              "Hello! I'm the HydroGeo Assistant powered by Agno and Gemini. I can help you understand environmental and hydrological data. What would you like to know?",
          },
        ]);
      }

      try {
        const response = await fetch('/api/chatbot/initialize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            context: 'hydrogeo_dataset',
            model: selectedModel,
            use_agno: true,
            session_id: sessionId
          }),
        });

        if (response.ok) {
          const data = await response.json();
          
          // Only update chat history if it's a new session or we have a welcome message
          if (!sessionId || data.welcome_message) {
            setChatHistory([
              {
                type: 'bot',
                content:
                  data.welcome_message ||
                  "Hello! I'm the HydroGeo Assistant powered by Agno and Gemini. I can help you understand environmental and hydrological data. What would you like to know?",
              },
            ]);
          }
          
          // Save the session ID
          if (data.session_id) {
            setSessionId(data.session_id);
          }
          
          setAgnoStatus({ connected: true, error: null });
        } else {
          console.warn('Server returned non-OK status for chatbot initialization');
          setAgnoStatus({ connected: false, error: 'Connection failed' });
        }
      } catch (error) {
        console.error('Error initializing chatbot:', error);
        setAgnoStatus({ connected: false, error: error.message });
      }
    } finally {
      setIsLoading(false);
    }
  }, [selectedModel, sessionId]);

  useEffect(() => {
    // Only initialize once when component mounts
    if (!initializedRef.current) {
      initializeAgent();
      initializedRef.current = true;
    }
  }, [initializeAgent]);
  
  // Only reinitialize when the model actually changes
  useEffect(() => {
    if (initializedRef.current && sessionId) {
      initializeAgent();
    }
  }, [selectedModel, initializeAgent, sessionId]);

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    // Add user message to chat history
    const userMessage = { type: 'user', content: message };
    setChatHistory((prev) => [...prev, userMessage]);
    setIsLoading(true);

    const currentMessage = message;
    setMessage(''); // Clear input field immediately after submission

    try {
      const response = await fetch('/api/chatbot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentMessage,
          context: 'hydrogeo_dataset',
          model: selectedModel,
          use_agno: true,
          session_id: sessionId
        }),
      });

      let botResponse;
      if (response.ok) {
        const data = await response.json();
        botResponse = data.response;
        setAgnoStatus({ connected: true, error: null });
      } else {
        console.error('Error response from server:', response.status);
        botResponse =
          'Sorry, I encountered an error while processing your request. Please try again.';
        setAgnoStatus({ connected: false, error: 'Request failed' });
      }

      // Add chatbot response to chat history
      setChatHistory((prev) => [...prev, { type: 'bot', content: botResponse }]);
    } catch (error) {
      console.error('Error sending message to chatbot:', error);
      setChatHistory((prev) => [
        ...prev,
        {
          type: 'bot',
          content:
            'Sorry, there was an error connecting to the assistant. Please check your network connection and try again.',
        },
      ]);
      setAgnoStatus({ connected: false, error: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelChange = (e) => {
    const newModel = e.target.value;
    if (newModel !== selectedModel) {
      setSelectedModel(newModel);
      // Clear session ID to force a new conversation
      setSessionId(null);
      setChatHistory([]);
    }
  };

  // Helper function to determine if a message might contain markdown
  const containsMarkdown = (content) => {
    const markdownPatterns = [
      /\*\*(.*?)\*\*/g,  // Bold
      /\*(.*?)\*/g,      // Italic
      /```([\s\S]*?)```/g, // Code blocks
      /`([^`]+)`/g,      // Inline code
      /\[(.*?)\]\((.*?)\)/g, // Links
      /#{1,6}\s.+/g,     // Headers
      /(-|\*|\+)\s.+/g,  // Lists
      /\n\n/g            // Paragraphs
    ];
    
    return markdownPatterns.some(pattern => pattern.test(content));
  };

  // Helper to render message content with basic markdown support
  const renderMessageContent = (content) => {
    if (!containsMarkdown(content)) {
      return <span>{content}</span>;
    }

    // Process markdown-like patterns
    let formattedContent = content;
    
    // Convert code blocks
    formattedContent = formattedContent.replace(/```([\s\S]*?)```/g, (match, code) => {
      return `<pre style="background-color:#2d2d2d; padding:1rem; border-radius:5px; overflow-x:auto; color:#e6e6e6;">${code}</pre>`;
    });
    
    // Convert inline code
    formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code style="background-color:#2d2d2d; padding:0.2rem; border-radius:3px; color:#e6e6e6;">$1</code>');
    
    // Convert bold
    formattedContent = formattedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert italic
    formattedContent = formattedContent.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert links
    formattedContent = formattedContent.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" style="color:#ff8500; text-decoration:underline;">$1</a>');
    
    // Convert headings
    formattedContent = formattedContent.replace(/^# (.*?)$/gm, '<h1 style="font-size:1.5rem; margin:0.5rem 0;">$1</h1>');
    formattedContent = formattedContent.replace(/^## (.*?)$/gm, '<h2 style="font-size:1.3rem; margin:0.5rem 0;">$1</h2>');
    formattedContent = formattedContent.replace(/^### (.*?)$/gm, '<h3 style="font-size:1.1rem; margin:0.5rem 0;">$1</h3>');
    
    // Convert lists (simple)
    formattedContent = formattedContent.replace(/^- (.*?)$/gm, '• $1<br/>');
    
    // Convert paragraphs
    formattedContent = formattedContent.replace(/\n\n/g, '<br/><br/>');
    formattedContent = formattedContent.replace(/\n/g, '<br/>');
    
    return <div dangerouslySetInnerHTML={{ __html: formattedContent }} />;
  };

  // Helper to render collapsible section
  const CollapsibleSection = ({ title, icon, isExpanded, setExpanded, children }) => (
    <div style={{ 
      marginBottom: '0.5rem',
      backgroundColor: '#2a2a2a',
      borderRadius: '8px',
      overflow: 'hidden',
      border: '1px solid #444'
    }}>
      <div 
        style={{ 
          padding: '0.7rem 1rem',
          display: 'flex', 
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          backgroundColor: '#333',
          borderBottom: isExpanded ? '1px solid #444' : 'none'
        }}
        onClick={() => setExpanded(!isExpanded)}
      >
        <h3 style={{ margin: 0, fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FontAwesomeIcon icon={icon} style={{ color: '#ff8500' }} />
          {title}
        </h3>
        <FontAwesomeIcon 
          icon={isExpanded ? faChevronUp : faChevronDown} 
          style={{ fontSize: '0.8rem', color: '#999' }}
        />
      </div>
      {isExpanded && (
        <div style={{ padding: '0.8rem 1rem' }}>
          {children}
        </div>
      )}
    </div>
  );

  return (
    <HydroGeoContainer style={{ 
      overflow: 'hidden'
    }}>
      <HydroGeoHeader style={{ 
        padding: '0.75rem', 
        marginBottom: '0.75rem', 
        flexShrink: 0,
        minHeight: 'auto'
      }}>
        <h1 style={{ fontSize: '1.8rem', margin: 0 }}>
          <FontAwesomeIcon icon={faRobot} style={{ marginRight: '0.8rem' }} />
          HydroGeo Assistant
          <span style={{ 
            fontSize: '0.7em', 
            backgroundColor: '#4a5568', 
            color: 'white', 
            padding: '0.2rem 0.5rem', 
            borderRadius: '0.5rem', 
            marginLeft: '0.8rem', 
            verticalAlign: 'middle' 
          }}>
            Powered by Agno
          </span>
        </h1>
      </HydroGeoHeader>

      <div style={{ 
        display: 'flex', 
        gap: '1.5rem', 
        height: 'calc(100vh - 120px)',
        maxHeight: 'calc(100vh - 120px)',
        overflow: 'hidden'
      }}>
        {/* Sidebar with collapsible elements */}
        <div style={{ 
          width: '280px', 
          overflowY: 'auto',
          overflowX: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
          flexShrink: 0,
          paddingRight: '0.5rem'
        }}>
          <CollapsibleSection 
            title="About the Assistant" 
            icon={faInfo} 
            isExpanded={aboutExpanded} 
            setExpanded={setAboutExpanded}
          >
            <p style={{ fontSize: '0.85rem', color: '#ccc', margin: 0 }}>
              The HydroGeo Assistant is powered by <strong>Agno</strong> and <strong>Gemini</strong>, offering 
              advanced AI capabilities for environmental data analysis. You can ask questions about data 
              sources, methodologies, analysis techniques, and more.
            </p>
            
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              marginTop: '0.8rem', 
              padding: '0.5rem', 
              backgroundColor: agnoStatus.connected ? 'rgba(72, 187, 120, 0.1)' : 'rgba(229, 62, 62, 0.1)', 
              borderRadius: '0.5rem',
              fontSize: '0.8rem'
            }}>
              <FontAwesomeIcon 
                icon={agnoStatus.connected ? faCheckCircle : agnoStatus.error ? faExclamationTriangle : faSpinner} 
                style={{ 
                  color: agnoStatus.connected ? '#48bb78' : '#e53e3e', 
                  marginRight: '0.5rem',
                  ...((!agnoStatus.connected && !agnoStatus.error) && { animation: 'spin 1s linear infinite' })
                }} 
              />
              <span style={{ color: agnoStatus.connected ? '#2f855a' : '#c53030' }}>
                {agnoStatus.connected ? 'Connected to Agno' : agnoStatus.error ? `Connection error: ${agnoStatus.error}` : 'Connecting to Agno...'}
              </span>
            </div>
          </CollapsibleSection>

          <CollapsibleSection 
            title="Sample Questions" 
            icon={faQuestionCircle} 
            isExpanded={samplesExpanded} 
            setExpanded={setSamplesExpanded}
          >
            <ul style={{ color: '#ccc', paddingLeft: '1.2rem', margin: '0', fontSize: '0.85rem' }}>
              <li>What climate data sources are available in the HydroGeo dataset?</li>
              <li>How can I interpret PRISM precipitation data?</li>
              <li>What is the difference between LOCA and CMIP climate projections?</li>
              <li>How do I create a report for my area of interest?</li>
            </ul>
          </CollapsibleSection>

          <CollapsibleSection 
            title="Model Settings" 
            icon={faCog} 
            isExpanded={settingsExpanded} 
            setExpanded={setSettingsExpanded}
          >
            <FormGroup style={{ margin: 0 }}>
              <InputField style={{ marginBottom: '0.5rem' }}>
                <select 
                  value={selectedModel} 
                  onChange={handleModelChange}
                  style={{ 
                    padding: '0.6rem 0.8rem',
                    backgroundColor: '#2a2a2a',
                    color: 'white',
                    border: '1px solid #444',
                    borderRadius: '0.5rem',
                    width: '100%',
                    fontSize: '0.85rem'
                  }}
                >
                  {availableModels.map((model) => (
                    <option key={model.id} value={model.id} disabled={model.disabled}>
                      {model.name} {model.disabled ? '(Coming Soon)' : ''}
                    </option>
                  ))}
                </select>
              </InputField>
              <p style={{ fontSize: '0.8rem', color: '#999', margin: '0.3rem 0 0 0' }}>
                Currently using: <strong>{selectedModel}</strong> by {availableModels.find(m => m.id === selectedModel)?.provider}
              </p>
            </FormGroup>
          </CollapsibleSection>
        </div>

        {/* Main chat area - takes most of the screen */}
        <ChatContainer style={{ 
          flexGrow: 1, 
          marginTop: 0,
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          maxHeight: '100%',
          overflow: 'hidden'
        }}>
          <ChatHeader style={{ 
            padding: '0.8rem',
            flexShrink: 0
          }}>
            <h2>
              <FontAwesomeIcon icon={faRobot} className="icon" />
              Chat with the Assistant
            </h2>
          </ChatHeader>

          <ChatMessagesContainer style={{ 
            flex: 1,
            overflowY: 'auto',
            overflowX: 'hidden',
            maxHeight: 'calc(100% - 75px)'
          }}>
            <MessageList>
              {chatHistory.map((chat, index) => (
                <MessageBubble key={index} className={chat.type}>
                  {renderMessageContent(chat.content)}
                </MessageBubble>
              ))}
              {isLoading && (
                <ThinkingIndicator>
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </ThinkingIndicator>
              )}
            </MessageList>
          </ChatMessagesContainer>

          <ChatInputContainer 
            onSubmit={handleChatSubmit}
            style={{ flexShrink: 0 }}
          >
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Ask a question about environmental data..."
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !message.trim()}>
              <FontAwesomeIcon icon={faPaperPlane} className="icon" />
            </button>
          </ChatInputContainer>
        </ChatContainer>
      </div>
    </HydroGeoContainer>
  );
};

export default HydroGeoAssistant; 