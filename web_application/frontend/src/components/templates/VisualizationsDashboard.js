import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import VisualizationForm from '../forms/Visualization';

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
    <main className="container my-5">
      <h1 className="text-center mb-4">Visualizations Dashboard</h1>
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
        className={`mt-5 ${showResults ? '' : 'd-none'}`}
        aria-labelledby="results-title"
      >
        <h2 id="results-title" className="text-center mb-4">
          Spatiotemporal Animations (GIFs)
        </h2>
        <div id="gif_container" className="d-flex flex-wrap justify-content-center gap-4">
          {visualizationResults && visualizationResults.length > 0 ? (
            visualizationResults.map((gif, idx) => (
              <div key={idx} className="gif-wrapper">
                <img src={gif} alt={`Animation ${idx + 1}`} className="img-fluid" />
              </div>
            ))
          ) : (
            <p>No visualizations available.</p>
          )}
        </div>
      </section>
    </main>
  );
};

export default VisualizationsDashboardTemplate;
