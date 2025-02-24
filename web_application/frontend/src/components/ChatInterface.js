import React, { useState } from 'react';
import styled from 'styled-components';

const ChatContainer = styled.div`
  width: 100%;
  margin-top: 20px;
  border: 1px solid #ccc;
  border-radius: 8px;
  overflow: hidden;
`;

const ChatMessages = styled.div`
  height: 300px;
  overflow-y: auto;
  padding: 15px;
  background: #f5f5f5;
`;

const Message = styled.div`
  margin: 10px 0;
  padding: 10px;
  border-radius: 8px;
  max-width: 80%;
  ${({ $messageType }) =>
    $messageType === 'bot'
      ? `
    background: #e3f2fd;
    margin-right: auto;
  `
      : `
    background: #e8f5e9;
    margin-left: auto;
  `}
`;

const CHATBOT_API_URL = 'http://localhost:5000/api/chatbot'; // Now calls Flask instead of Ollama directly

const ChatInterface = ({ dataResults }) => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const generateChatbotResponse = async (data) => {
    setIsLoading(true);
    try {
      const response = await fetch(CHATBOT_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'deepseek-r1:70b',
          prompt: `Analyze this environmental data summary: ${JSON.stringify(data).slice(0, 500)}...`,
          stream: false,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      return result.response || "Sorry, I couldn't analyze the data.";
    } catch (error) {
      console.error('Error calling chatbot API:', error);
      return `Error: ${error.message}`;
    } finally {
      setIsLoading(false);
    }
  };

  React.useEffect(() => {
    if (dataResults) {
      setMessages((prev) => [
        ...prev,
        {
          text: 'New data received:',
          messageType: 'user',
          data: dataResults,
        },
      ]);

      generateChatbotResponse(dataResults).then((response) => {
        setMessages((prev) => [
          ...prev,
          {
            text: response,
            messageType: 'bot',
          },
        ]);
      });
    }
  }, [dataResults]);

  return (
    <ChatContainer>
      <ChatMessages>
        {messages.map((message, index) => (
          <Message key={index} $messageType={message.messageType}>
            <div>{message.text}</div>
            {message.data && (
              <pre style={{ fontSize: '0.8em', marginTop: '10px' }}>
                {JSON.stringify(message.data, null, 2)}
              </pre>
            )}
          </Message>
        ))}
        {isLoading && <Message $messageType="bot">Analyzing data...</Message>}
      </ChatMessages>
    </ChatContainer>
  );
};

export default ChatInterface;
