import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import VisualizationForm from '../forms/Visualization.js';
import { Box } from '@mui/material';
import {
  Body,
  FormContainer,
  PageTitle,
  SectionTitle,
  DescriptionContainer,
  GifContainer,
  GifWrapper,
  NoResults,
  DownloadButton,
  Collapsible,
  CollapsibleHeader,
  CollapsibleContent,
  StyledCircularProgress,
  ErrorMessage,
  LoadingContainer,
} from '../../styles/Visualization.tsx';

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
  const [isCollapsed, setIsCollapsed] = useState(true);

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
    setLoading(true); // Show loading state

    console.log('POST /visualizations params:', {
      NAME: selectedWatershed,
      ver: ensemble,
      variable: selectedVariables.join(','),
    });

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
      })
      .finally(() => {
        setLoading(false); // Hide loading state
      });
  };

  return (
    <Body>
      <FormContainer maxWidth="lg">
        <Box display="flex" justifyContent="center" width="100%">
          <PageTitle variant="h1">SWAT+ Model Visualizations</PageTitle>
        </Box>

        {/* Description Section */}
        <DescriptionContainer>
          <p>
            This dashboard provides access to the **latest available visualizations** of SWAT+
            models. Users can explore **spatiotemporal simulations** of hydrological parameters such
            as **streamflow, soil moisture, groundwater recharge, and evapotranspiration** across
            different watersheds.
          </p>
          <p>
            Select a **watershed, model ensemble, and variables** to generate visualizations. You
            can also **download the generated animations** for further analysis.
          </p>
        </DescriptionContainer>

        {/* Additional Information (Collapsible) */}
        <Collapsible>
          <CollapsibleHeader onClick={() => setIsCollapsed(!isCollapsed)}>
            {isCollapsed ? '▶' : '▼'} About SWAT+ Model Data
          </CollapsibleHeader>
          {!isCollapsed && (
            <CollapsibleContent>
              <p>
                The **Soil & Water Assessment Tool Plus (SWAT+)** is a watershed-scale model used to
                simulate **water balance, nutrient transport, and land use impacts** on hydrological
                processes. This visualization tool helps researchers and water managers analyze
                SWAT+ simulations interactively.
              </p>
            </CollapsibleContent>
          )}
        </Collapsible>

        {/* Form Section */}
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
            <StyledCircularProgress size={40} />
          </LoadingContainer>
        )}

        {/* Visualization Results */}
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
                  <DownloadButton href={gif} download={`Visualization_${idx + 1}.gif`}>
                    Download
                  </DownloadButton>
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
