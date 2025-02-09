import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import VisualizationForm from '../forms/Visualization.js';
import {
  Body,
  FormContainer,
  PageTitle,
  SectionTitle,
  GifContainer,
  GifWrapper,
  NoResults,
} from '../../styles/VisualizationsDashboard.tsx';
import { Box } from '@mui/material';

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

  // Fetch dropdown options on component mount
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await axios.get('/get_options');
        console.log('GET /get_options response:', response.data); // Logging after GET request
        setWatersheds(response.data.names);
        setAvailableVariables(response.data.variables);
      } catch (error) {
        if (error.response && error.response.status === 401) {
          navigate('/login');
        } else {
          console.error('Error fetching options:', error);
        }
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

    // Fetch visualization results from backend
    console.log('POST /visualizations params:', {
      NAME: selectedWatershed,
      ver: ensemble,
      variable: selectedVariables.join(','),
    }); // Logging before POST request
    axios
      .get('/visualizations', {
        params: {
          NAME: selectedWatershed,
          ver: ensemble,
          variable: selectedVariables.join(','),
        },
      })
      .then((response) => {
        setVisualizationResults(response.data.gif_files);
        setShowResults(true);
      })
      .catch((error) => {
        console.error('Error fetching visualizations:', error);
        setErrorMessage('Failed to fetch visualizations.');
        setShowResults(false);
      });
  };

  return (
    <Body>
      <FormContainer maxWidth="lg">
        <Box display="flex" justifyContent="center" width="100%">
          <PageTitle variant="h1">Visualizations Dashboard</PageTitle>
        </Box>
        <section>
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
        </section>

        <section
          id="visualization_results"
          style={{ display: showResults ? 'block' : 'none' }}
          aria-labelledby="results-title"
        >
          <SectionTitle variant="h2" id="results-title">
            Spatiotemporal Animations (GIFs)
          </SectionTitle>
          <GifContainer>
            {visualizationResults && visualizationResults.length > 0 ? (
              visualizationResults.map((gif, idx) => (
                <GifWrapper key={idx}>
                  <img src={gif} alt={`Animation ${idx + 1}`} />
                </GifWrapper>
              ))
            ) : (
              <NoResults>No visualizations available.</NoResults>
            )}
          </GifContainer>
        </section>
      </FormContainer>
    </Body>
  );
};

export default VisualizationsDashboardTemplate;
