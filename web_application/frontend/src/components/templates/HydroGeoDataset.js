import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faSearch,
  faMapMarkerAlt,
  faLayerGroup,
  faDatabase,
  faRobot,
  faMapMarkedAlt,
  faInfoCircle,
  faPaperPlane,
  faChartBar,
  faFilter,
  faCompass,
  faMapPin,
  faSpinner,
} from '@fortawesome/free-solid-svg-icons';

import MapComponent from '../MapComponent';
import HydroGeoDatasetForm from '../forms/HydroGeoDataset';
import * as webMercatorUtils from '@arcgis/core/geometry/support/webMercatorUtils';

import {
  HydroGeoContainer,
  HydroGeoHeader,
  ContentLayout,
  QuerySidebar,
  MapContainer,
  ChatContainer,
  ChatHeader,
  ChatMessagesContainer,
  MessageBubble,
  MessageList,
  ChatInputContainer,
  ResultsContainer,
  ThinkingIndicator,
} from '../../styles/HydroGeoDataset.tsx';

const HydroGeoDataset = () => {
  const [formData, setFormData] = useState({
    min_latitude: '',
    max_latitude: '',
    min_longitude: '',
    max_longitude: '',
    variable: '',
    subvariable: '',
    geometry: null,
  });
  const [availableVariables, setAvailableVariables] = useState([]);
  const [availableSubvariables, setAvailableSubvariables] = useState([]);
  const [data, setData] = useState(null);

  // Chatbot related states
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await fetch('/hydro_geo_dataset');
        const data = await response.json();
        setAvailableVariables(data.variables);
      } catch (error) {
        console.error('Error fetching options:', error);
      }
    };
    fetchOptions();
  }, []);

  useEffect(() => {
    const fetchSubvariables = async () => {
      if (!formData.variable) return;
      try {
        const response = await fetch(`/hydro_geo_dataset?variable=${formData.variable}`);
        const data = await response.json();
        setAvailableSubvariables(data.subvariables);
      } catch (error) {
        console.error('Error fetching subvariables:', error);
      }
    };
    fetchSubvariables();
  }, [formData.variable]);

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

  const handleChange = ({ target: { name, value } }) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleGeometryChange = (geom) => {
    if (!geom) return;
    let convertedGeometry = { ...geom };
    if (geom.rings) {
      convertedGeometry.rings = geom.rings.map((ring) =>
        ring.map((coord) => {
          const geographicPoint = webMercatorUtils.xyToLngLat(coord[0], coord[1]);
          return [geographicPoint[0].toFixed(6), geographicPoint[1].toFixed(6)];
        }),
      );
    }
    setFormData((prev) => ({ ...prev, geometry: convertedGeometry }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('Submitting data:', JSON.stringify(formData, null, 2));
    try {
      setIsLoading(true);
      const response = await fetch('/hydro_geo_dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setIsLoading(false);
    }
  };

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
          <FontAwesomeIcon icon={faDatabase} style={{ marginRight: '0.8rem' }} />
          HydroGeoDataset Explorer
        </h1>
        <p>
          Access high-resolution hydrological, environmental, and climate data including PRISM,
          LOCA2, and Wellogic records through a simple interface. Select your area of interest and
          data variables.
        </p>
      </HydroGeoHeader>

      <ContentLayout>
        <QuerySidebar>
          <HydroGeoDatasetForm
            formData={formData}
            handleChange={handleChange}
            handleSubmit={handleSubmit}
            availableVariables={availableVariables}
            availableSubvariables={availableSubvariables}
            isLoading={isLoading}
          />
        </QuerySidebar>

        <MapContainer>
          <MapComponent setFormData={setFormData} onGeometryChange={handleGeometryChange} />
        </MapContainer>
      </ContentLayout>

      {data && (
        <ResultsContainer>
          <h2>
            <FontAwesomeIcon icon={faChartBar} className="icon" />
            Query Results
          </h2>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </ResultsContainer>
      )}

      <ChatContainer>
        <ChatHeader>
          <h2>
            <FontAwesomeIcon icon={faRobot} className="icon" />
            HydroGeo Assistant
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
            placeholder="Ask a question about the data..."
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

export default HydroGeoDataset;
