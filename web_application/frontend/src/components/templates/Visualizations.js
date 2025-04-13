import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import VisualizationForm from '../forms/Visualization.js';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Button,
  Grid,
  CircularProgress,
  Card,
  CardMedia,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { styled } from '@mui/system';

// Styled Components
const VisualizationContainer = styled(Box)({
  margin: '2rem auto',
  padding: '2rem',
  borderRadius: '16px',
  boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)',
  backgroundColor: '#444e5e',
  color: 'white',
  textAlign: 'center',
});

const VisualizationTitle = styled(Typography)({
  fontSize: '2.5rem',
  fontWeight: 'bold',
  marginBottom: '1.5rem',
});

const ErrorMessage = styled(Typography)({
  color: '#ff3333',
  fontWeight: 'bold',
  marginTop: '1rem',
});

const LoadingContainer = styled(Box)({
  display: 'flex',
  justifyContent: 'center',
  marginTop: '1.5rem',
});

const DownloadButton = styled(Button)({
  marginTop: '0.5rem',
  backgroundColor: '#e67500',
  color: 'white',
  '&:hover': {
    backgroundColor: '#ff5722',
  },
});

// Component
const VisualizationsDashboardTemplate = () => {
  const navigate = useNavigate();
  // Form states
  const [watersheds, setWatersheds] = useState([]);
  const [selectedWatershed, setSelectedWatershed] = useState('');
  const [ensemble, setEnsemble] = useState('');
  const [availableVariables, setAvailableVariables] = useState([]);
  const [selectedVariables, setSelectedVariables] = useState([]);
  // UI states
  const [errorMessage, setErrorMessage] = useState('');
  const [visualizationResults, setVisualizationResults] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [loading, setLoading] = useState(false);

  // Fetch dropdown options on component mount
  useEffect(() => {
    const fetchOptions = async () => {
      setLoading(true);
      try {
        const response = await axios.get('/api/get_options');
        console.log('GET /api/get_options response:', response.data);

        // Safely set the data with defaults
        setWatersheds(response.data.names || []);
        setAvailableVariables(response.data.variables || []);
      } catch (error) {
        console.error('Error fetching options:', error);
        if (error.response && error.response.status === 401) {
          navigate('/login');
        } else {
          setErrorMessage('Failed to load watershed and variable options. Please try again later.');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchOptions();
  }, [navigate]);

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();

    // Basic validation
    if (!selectedWatershed || !ensemble || selectedVariables.length === 0) {
      setErrorMessage('Please fill in all required fields.');
      setShowResults(false);
      return;
    }

    setErrorMessage('');
    setLoading(true);

    console.log('GET /api/visualizations params:', {
      NAME: selectedWatershed,
      ver: ensemble,
      variable: selectedVariables.join(','),
    });

    axios
      .get('/api/visualizations', {
        params: {
          NAME: selectedWatershed,
          ver: ensemble,
          variable: selectedVariables.join(','),
        },
      })
      .then((response) => {
        setVisualizationResults(response.data.gif_files || []);
        setShowResults(true);

        // Handle warnings if present
        if (response.data.warnings) {
          setErrorMessage(response.data.warnings);
        }
      })
      .catch((error) => {
        console.error('Error fetching visualizations:', error);
        if (error.response && error.response.data && error.response.data.error) {
          setErrorMessage(error.response.data.error);
        } else {
          setErrorMessage('Failed to fetch visualizations. Please try again later.');
        }
        setShowResults(false);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <VisualizationContainer>
      <VisualizationTitle variant="h1">Visualizations Dashboard</VisualizationTitle>

      {/* Collapsible Description */}
      <Accordion
        sx={{ backgroundColor: '#333', color: 'white', borderRadius: '8px', marginBottom: '1rem' }}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: 'white' }} />}>
          <Typography variant="h6">Description</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            This dashboard provides access to the latest available visualizations of calibrated
            SWAT+ models. Users can select a watershed, model ensemble, and variables to generate
            visualizations.
          </Typography>
        </AccordionDetails>
      </Accordion>

      {/* Visualization Form */}
      <VisualizationForm
        watersheds={watersheds}
        selectedWatershed={selectedWatershed}
        setSelectedWatershed={setSelectedWatershed}
        ensemble={ensemble}
        setEnsemble={setEnsemble}
        availableVariables={availableVariables}
        selectedVariables={selectedVariables}
        setSelectedVariables={setSelectedVariables}
        handleSubmit={handleSubmit}
        errorMessage={errorMessage}
      />

      {/* Error Message */}
      {errorMessage && <ErrorMessage>{errorMessage}</ErrorMessage>}

      {/* Loading Indicator */}
      {loading && (
        <LoadingContainer>
          <CircularProgress size={40} sx={{ color: '#ff5722' }} />
        </LoadingContainer>
      )}

      {/* Results Section */}
      {showResults && (
        <Box sx={{ marginTop: '2rem' }}>
          <Grid container spacing={2} justifyContent="center">
            {visualizationResults.length > 0 ? (
              visualizationResults.map((gif, idx) => (
                <Grid item xs={12} sm={6} md={4} key={idx}>
                  <Card sx={{ backgroundColor: '#222', borderRadius: '8px', padding: '0.5rem' }}>
                    <CardMedia
                      component="img"
                      image={gif}
                      alt={`Visualization ${idx + 1}`}
                      sx={{ borderRadius: '8px', maxHeight: '250px', objectFit: 'contain' }}
                    />
                    <DownloadButton href={gif} download={`Visualization_${idx + 1}.gif`} fullWidth>
                      Download
                    </DownloadButton>
                  </Card>
                </Grid>
              ))
            ) : (
              <Typography variant="h6" sx={{ color: 'white', marginTop: '1rem' }}>
                No visualizations available.
              </Typography>
            )}
          </Grid>
        </Box>
      )}
    </VisualizationContainer>
  );
};

export default VisualizationsDashboardTemplate;
