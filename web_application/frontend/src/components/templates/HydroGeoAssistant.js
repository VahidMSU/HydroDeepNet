import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faRobot,
  faPaperPlane,
  faInfo,
  faQuestionCircle,
  faCog,
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
  FormGroup,
} from '../../styles/HydroGeoDataset.tsx';

const HydroGeoAssistant = () => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  //const [selectedModel] = useState('llama3:latest');
  const [aboutExpanded, setAboutExpanded] = useState(false);
  const [samplesExpanded, setSamplesExpanded] = useState(false);
  const [settingsExpanded, setSettingsExpanded] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const initializedRef = useRef(false);

  const initializeAgent = useCallback(async () => {
    try {
      setIsLoading(true);

      if (!sessionId) {
        setChatHistory([
          {
            type: 'bot',
            content:
              "Hello! I'm the HydroGeo Assistant powered by Ollama. I can help you understand environmental and hydrological data. What would you like to know?",
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
            model: 'llama3:latest',
            session_id: sessionId,
          }),
        });

        if (response.ok) {
          const data = await response.json();

          if (!sessionId || data.welcome_message) {
            setChatHistory([
              {
                type: 'bot',
                content:
                  data.welcome_message ||
                  "Hello! I'm the HydroGeo Assistant powered by Ollama. I can help you understand environmental and hydrological data. What would you like to know?",
              },
            ]);
          }

          if (data.session_id) {
            setSessionId(data.session_id);
          }
        } else {
          console.warn('Server returned non-OK status for chatbot initialization');
        }
      } catch (error) {
        console.error('Error initializing chatbot:', error);
      }
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    if (!initializedRef.current) {
      initializeAgent();
      initializedRef.current = true;
    }
  }, [initializeAgent]);

  useEffect(() => {
    if (initializedRef.current && sessionId) {
      initializeAgent();
    }
  }, [initializeAgent, sessionId]);

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    const userMessage = { type: 'user', content: message };
    setChatHistory((prev) => [...prev, userMessage]);
    setIsLoading(true);

    const currentMessage = message;
    setMessage('');

    try {
      const response = await fetch('/api/chatbot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentMessage,
          context: 'hydrogeo_dataset',
          model: 'llama3:latest',
          session_id: sessionId,
        }),
      });

      let botResponse;
      if (response.ok) {
        const data = await response.json();
        botResponse = data.response;
      } else {
        console.error('Error response from server:', response.status);
        botResponse =
          'Sorry, I encountered an error while processing your request. Please try again.';
      }

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
    } finally {
      setIsLoading(false);
    }
  };

  const containsMarkdown = (content) => {
    const markdownPatterns = [
      /\*\*(.*?)\*\*/g,
      /\*(.*?)\*/g,
      /```([\s\S]*?)```/g,
      /`([^`]+)`/g,
      /\[(.*?)\]\((.*?)\)/g,
      /#{1,6}\s.+/g,
      /(-|\*|\+)\s.+/g,
      /\n\n/g,
    ];

    return markdownPatterns.some((pattern) => pattern.test(content));
  };

  const renderMessageContent = (content) => {
    if (!containsMarkdown(content)) {
      return <span>{content}</span>;
    }

    let formattedContent = content;

    formattedContent = formattedContent.replace(/```([\s\S]*?)```/g, (match, code) => {
      return `<pre style="background-color:#2d2d2d; padding:1rem; border-radius:5px; overflow-x:auto; color:#e6e6e6;">${code}</pre>`;
    });

    formattedContent = formattedContent.replace(
      /`([^`]+)`/g,
      '<code style="background-color:#2d2d2d; padding:0.2rem; border-radius:3px; color:#e6e6e6;">$1</code>',
    );

    formattedContent = formattedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    formattedContent = formattedContent.replace(/\*(.*?)\*/g, '<em>$1</em>');

    formattedContent = formattedContent.replace(
      /\[(.*?)\]\((.*?)\)/g,
      '<a href="$2" target="_blank" style="color:#ff5722; text-decoration:underline;">$1</a>',
    );

    formattedContent = formattedContent.replace(
      /^# (.*?)$/gm,
      '<h1 style="font-size:1.5rem; margin:0.5rem 0;">$1</h1>',
    );
    formattedContent = formattedContent.replace(
      /^## (.*?)$/gm,
      '<h2 style="font-size:1.3rem; margin:0.5rem 0;">$1</h2>',
    );
    formattedContent = formattedContent.replace(
      /^### (.*?)$/gm,
      '<h3 style="font-size:1.1rem; margin:0.5rem 0;">$1</h3>',
    );

    formattedContent = formattedContent.replace(/^- (.*?)$/gm, '• $1<br/>');

    formattedContent = formattedContent.replace(/\n\n/g, '<br/><br/>');
    formattedContent = formattedContent.replace(/\n/g, '<br/>');

    return <div dangerouslySetInnerHTML={{ __html: formattedContent }} />;
  };

  const CollapsibleSection = ({ title, icon, isExpanded, setExpanded, children }) => (
    <div
      style={{
        marginBottom: '0.5rem',
        backgroundColor: '#2b2b2c',
        borderRadius: '8px',
        overflow: 'hidden',
        border: '1px solid #3f3f45',
      }}
    >
      <div
        style={{
          padding: '0.7rem 1rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          backgroundColor: '#333',
          borderBottom: isExpanded ? '1px solid #3f3f45' : 'none',
        }}
        onClick={() => setExpanded(!isExpanded)}
      >
        <h3
          style={{
            margin: 0,
            fontSize: '0.95rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <FontAwesomeIcon icon={icon} style={{ color: '#ff5722' }} />
          {title}
        </h3>
        <FontAwesomeIcon
          icon={isExpanded ? faChevronUp : faChevronDown}
          style={{ fontSize: '0.8rem', color: '#999' }}
        />
      </div>
      {isExpanded && <div style={{ padding: '0.8rem 1rem' }}>{children}</div>}
    </div>
  );

  return (
    <HydroGeoContainer
      style={{
        overflow: 'hidden',
        backgroundColor: '#1c1c1e',
        padding: 0,
        minHeight: '100vh',
        maxHeight: '100vh',
      }}
    >
      <HydroGeoHeader
        style={{
          padding: '0.75rem',
          marginBottom: '0.75rem',
          flexShrink: 0,
          minHeight: 'auto',
          background: '#2b2b2c',
          borderRadius: '10px',
          margin: '10px',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        }}
      >
        <h1 style={{ fontSize: '1.8rem', margin: 0 }}>
          <FontAwesomeIcon icon={faRobot} style={{ marginRight: '0.8rem', color: '#ff5722' }} />
          HydroGeo Assistant
        </h1>
      </HydroGeoHeader>

      <div
        style={{
          display: 'flex',
          gap: '1.5rem',
          height: 'calc(100vh - 120px)',
          maxHeight: 'calc(100vh - 120px)',
          overflow: 'hidden',
          padding: '0 10px 10px 10px',
        }}
      >
        <div
          style={{
            width: '280px',
            overflowY: 'auto',
            overflowX: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem',
            flexShrink: 0,
            paddingRight: '0.5rem',
            backgroundColor: '#2b2b2c',
            borderRadius: '10px',
            padding: '10px',
          }}
        >
          <CollapsibleSection
            title="About the Assistant"
            icon={faInfo}
            isExpanded={aboutExpanded}
            setExpanded={setAboutExpanded}
          >
            <p style={{ fontSize: '0.85rem', color: '#c5c5c8', margin: 0 }}>
              The HydroGeo Assistant is powered by <strong>Ollama</strong>, offering advanced AI
              capabilities for environmental data analysis. You can ask questions about data
              sources, methodologies, analysis techniques, and more.
            </p>
          </CollapsibleSection>

          <CollapsibleSection
            title="Sample Questions"
            icon={faQuestionCircle}
            isExpanded={samplesExpanded}
            setExpanded={setSamplesExpanded}
          >
            <ul
              style={{ color: '#c5c5c8', paddingLeft: '1.2rem', margin: '0', fontSize: '0.85rem' }}
            >
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
              <p style={{ fontSize: '0.85rem', color: '#c5c5c8', margin: 0 }}>
                This assistant is powered by <strong>Ollama</strong>, to provide the highest quality
                responses for environmental data analysis.
              </p>
              <p style={{ fontSize: '0.8rem', color: '#9e9e9e', margin: '0.8rem 0 0 0' }}>
                Currently using: <strong>llama3 & Llama3.2-Vision</strong> by Ollama
              </p>
            </FormGroup>
          </CollapsibleSection>
        </div>

        <ChatContainer
          style={{
            flexGrow: 1,
            marginTop: 0,
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            maxHeight: '100%',
            overflow: 'hidden',
            backgroundColor: '#2b2b2c',
            borderRadius: '10px',
          }}
        >
          <ChatHeader
            style={{
              padding: '0.8rem',
              flexShrink: 0,
            }}
          >
            <h2>
              <FontAwesomeIcon icon={faRobot} className="icon" />
              Chat with the Assistant
            </h2>
          </ChatHeader>

          <ChatMessagesContainer
            style={{
              flex: 1,
              overflowY: 'auto',
              overflowX: 'hidden',
              maxHeight: 'calc(100% - 75px)',
            }}
          >
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

          <ChatInputContainer onSubmit={handleChatSubmit} style={{ flexShrink: 0 }}>
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
