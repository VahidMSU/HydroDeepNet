import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faRobot,
  faPaperPlane,
  faInfo,
  faQuestionCircle,
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
  InfoCard,
} from '../../styles/HydroGeoDataset.tsx';

const HydroGeoAssistant = () => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const initializeAgent = async () => {
      try {
        setIsLoading(true);

        // First set a default welcome message in case the request fails
        setChatHistory([
          {
            type: 'bot',
            content:
              "Hello! I'm the HydroGeo Assistant. I can help you understand environmental and hydrological data. What would you like to know?",
          },
        ]);

        try {
          const response = await fetch('/api/chatbot/initialize', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ context: 'hydrogeo_dataset' }),
          });

          if (response.ok) {
            const data = await response.json();
            // Update with the server-provided message if available
            setChatHistory([
              {
                type: 'bot',
                content:
                  data.welcome_message ||
                  "Hello! I'm the HydroGeo Assistant. How can I help you with environmental and hydrological data today?",
              },
            ]);
          } else {
            console.warn('Server returned non-OK status for chatbot initialization');
            // Keep the default message already set
          }
        } catch (error) {
          console.error('Error initializing chatbot:', error);
        }
      } finally {
        setIsLoading(false);
      }
    };

    // Initialize the agent when the component mounts
    initializeAgent();
  }, []);

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
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <HydroGeoContainer>
      <HydroGeoHeader>
        <h1>
          <FontAwesomeIcon icon={faRobot} style={{ marginRight: '0.8rem' }} />
          HydroGeo Assistant
        </h1>
        <p>
          Ask questions about environmental and hydrological data. This AI assistant can help you
          understand data sources, interpret results, and guide you through using the HydroGeo
          dataset explorer.
        </p>
      </HydroGeoHeader>

      <InfoCard>
        <h3>
          <FontAwesomeIcon icon={faInfo} className="icon" />
          About the Assistant
        </h3>
        <p>
          The HydroGeo Assistant is an AI-powered tool designed to help you navigate and understand
          environmental data. You can ask questions about data sources, methodologies, analysis
          techniques, and more.
        </p>
      </InfoCard>

      <InfoCard>
        <h3>
          <FontAwesomeIcon icon={faQuestionCircle} className="icon" />
          Sample Questions
        </h3>
        <ul style={{ color: '#666', paddingLeft: '1.5rem', margin: '0.5rem 0' }}>
          <li>What climate data sources are available in the HydroGeo dataset?</li>
          <li>How can I interpret PRISM precipitation data?</li>
          <li>What is the difference between LOCA and CMIP climate projections?</li>
          <li>How do I create a report for my area of interest?</li>
        </ul>
      </InfoCard>

      <ChatContainer style={{ flexGrow: 1, marginTop: '2rem' }}>
        <ChatHeader>
          <h2>
            <FontAwesomeIcon icon={faRobot} className="icon" />
            Chat with the Assistant
          </h2>
        </ChatHeader>

        <ChatMessagesContainer>
          <MessageList>
            {chatHistory.map((chat, index) => (
              <MessageBubble key={index} className={chat.type}>
                {chat.content}
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

        <ChatInputContainer onSubmit={handleChatSubmit}>
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
    </HydroGeoContainer>
  );
};

export default HydroGeoAssistant; 