import React, { useState, useEffect } from 'react';
import MapComponent from '../MapComponent';
import HydroGeoDatasetForm from '../forms/HydroGeoDataset';
import * as webMercatorUtils from '@arcgis/core/geometry/support/webMercatorUtils';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Paper,
  List,
  ListItem,
  Divider,
} from '@mui/material';

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

  // Initialize the chatbot when component mounts
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
          // The default message is already set, just log the error
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
      const response = await fetch('/hydro_geo_dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error('Error fetching data:', error);
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
    <Box display="flex" flexDirection="column" p={3}>
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h2" gutterBottom sx={{ fontWeight: 'bold', color: 'white' }}>
          HydroGeoDataset Overview
        </Typography>
        <Typography variant="h7" sx={{ color: 'white' }}>
          HydroGeoDataset provides high-resolution hydrological, environmental, and climate data. It
          includes datasets such as PRISM, LOCA2, and Wellogic records.
        </Typography>
      </Box>

      <Box mt={3} display="flex" gap={3}>
        <Card sx={{ width: '30%', bgcolor: 'white', minHeight: '600px' }}>
          <CardContent>
            <Typography
              variant="h4"
              align="center"
              sx={{ fontWeight: 'bold', color: '#2b2b2c', m: 2 }}
            >
              Data Query
            </Typography>
            <HydroGeoDatasetForm
              formData={formData}
              handleChange={handleChange}
              handleSubmit={handleSubmit}
              availableVariables={availableVariables}
              availableSubvariables={availableSubvariables}
            />
          </CardContent>
        </Card>

        <Box flex={1}>
          <MapComponent setFormData={setFormData} onGeometryChange={handleGeometryChange} />
        </Box>
      </Box>

      {/* Chatbot Card */}
      <Card sx={{ mt: 3, p: 2, bgcolor: 'white' }}>
        <CardContent>
          <Typography
            variant="h4"
            align="center"
            sx={{ fontWeight: 'bold', color: '#2b2b2c', mb: 3 }}
          >
            HydroGeo Assistant
          </Typography>

          <Paper
            elevation={3}
            sx={{
              height: '300px',
              mb: 2,
              p: 2,
              overflowY: 'auto',
              bgcolor: '#f9f9f9',
            }}
          >
            <List>
              {chatHistory.length === 0 ? (
                <ListItem>
                  <Typography variant="body1" color="textSecondary" sx={{ fontStyle: 'italic' }}>
                    Initializing assistant...
                  </Typography>
                </ListItem>
              ) : (
                chatHistory.map((chat, index) => (
                  <React.Fragment key={index}>
                    <ListItem
                      sx={{
                        flexDirection: 'column',
                        alignItems: chat.type === 'user' ? 'flex-end' : 'flex-start',
                        mb: 1,
                      }}
                    >
                      <Paper
                        elevation={1}
                        sx={{
                          p: 1.5,
                          maxWidth: '80%',
                          bgcolor: chat.type === 'user' ? '#e3f2fd' : '#ffffff',
                          borderRadius: '8px',
                        }}
                      >
                        <Typography variant="body1">{chat.content}</Typography>
                      </Paper>
                    </ListItem>
                    {index < chatHistory.length - 1 && <Divider variant="middle" />}
                  </React.Fragment>
                ))
              )}
              {isLoading && (
                <ListItem sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                  <Paper
                    elevation={1}
                    sx={{ p: 1.5, maxWidth: '80%', bgcolor: '#ffffff', borderRadius: '8px' }}
                  >
                    <Typography variant="body1">Thinking...</Typography>
                  </Paper>
                </ListItem>
              )}
            </List>
          </Paper>

          <Box component="form" onSubmit={handleChatSubmit} sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Type your question here..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              sx={{
                bgcolor: 'white',
                '& .MuiOutlinedInput-root': {
                  '& fieldset': {
                    borderColor: '#ccc',
                  },
                  '&:hover fieldset': {
                    borderColor: '#ff8500',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: '#ff8500',
                  },
                },
              }}
            />
            <Button
              type="submit"
              variant="contained"
              sx={{
                bgcolor: '#ff8500',
                '&:hover': {
                  bgcolor: '#e67500',
                },
              }}
              disabled={isLoading || !message.trim()}
            >
              Send
            </Button>
          </Box>
        </CardContent>
      </Card>

      {data && (
        <Card sx={{ mt: 3, p: 2 }}>
          <Typography variant="h6">Results</Typography>
          <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        </Card>
      )}
    </Box>
  );
};

export default HydroGeoDataset;
