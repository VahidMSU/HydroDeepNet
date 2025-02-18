import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import VisualizationForm from '../forms/Visualization.js';
import {
  VisualizationContainer,
  VisualizationTitle,
  ContentWrapper,
  GifContainer,
  GifWrapper,
  DownloadButton,
  Collapsible,
  CollapsibleHeader,
  CollapsibleContent,
  ErrorMessage,
  LoadingContainer,
  StyledCircularProgress,
} from '../../styles/Visualizations.tsx';

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
    <VisualizationContainer>
      <VisualizationTitle>Visualizations Dashboard</VisualizationTitle>

      <ContentWrapper>
        <Collapsible>
          <CollapsibleHeader onClick={() => setIsCollapsed(!isCollapsed)}>
            {isCollapsed ? '▶' : '▼'} Description
          </CollapsibleHeader>
          {!isCollapsed && (
            <CollapsibleContent>
              <p>
                This dashboard provides access to the latest available visualizations of calibrated
                SWAT+ models. Users can select a watershed, model ensemble, and variables to
                generate visualizations.
              </p>
            </CollapsibleContent>
          )}
        </Collapsible>

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

        {errorMessage && <ErrorMessage>{errorMessage}</ErrorMessage>}

        {loading && (
          <LoadingContainer>
            <StyledCircularProgress size={40} />
          </LoadingContainer>
        )}

        {showResults && (
          <section>
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
                <p>No visualizations available.</p>
              )}
            </GifContainer>
          </section>
        )}
      </ContentWrapper>
    </VisualizationContainer>
  );
};

export default VisualizationsDashboardTemplate;
