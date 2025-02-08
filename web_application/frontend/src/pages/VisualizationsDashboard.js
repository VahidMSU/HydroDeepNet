// pages/VisualizationsDashboard.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import '../css/VisualizationsDashboard.css'; // Adjust the path if needed

const VisualizationsDashboard = () => {
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

  // Handle multiple selection change
  const handleVariableChange = (e) => {
    const { options } = e.target;
    const selected = [];
    for (let i = 0; i < options.length; i++) {
      if (options[i].selected) {
        selected.push(options[i].value);
      }
    }
    setSelectedVariables(selected);
  };

  return (
    <main className="container my-5">
      <h1 className="text-center mb-4">Visualizations Dashboard</h1>
      <section>
        <form id="visualization_form" className="form-container" onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="NAME" className="form-label">
                Watershed Name
              </label>
              <select
                name="NAME"
                id="NAME"
                className="form-select"
                required
                value={selectedWatershed}
                onChange={(e) => setSelectedWatershed(e.target.value)}
              >
                <option value="">Select a Watershed</option>
                {watersheds.map((w, idx) => (
                  <option key={idx} value={w}>
                    {w}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="ver" className="form-label">
                Ensemble
              </label>
              <input
                type="text"
                name="ver"
                id="ver"
                className="form-control"
                placeholder="Enter Ensemble Version"
                required
                value={ensemble}
                onChange={(e) => setEnsemble(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label htmlFor="variable" className="form-label">
                Variable Name
              </label>
              <select
                name="variable"
                id="variable"
                className="form-select"
                multiple
                value={selectedVariables}
                onChange={handleVariableChange}
              >
                {availableVariables.map((variable, idx) => (
                  <option key={idx} value={variable}>
                    {variable}
                  </option>
                ))}
              </select>
              <small id="variable-help" className="form-text text-muted">
                Hold Ctrl/Cmd to select multiple variables.
              </small>
            </div>
          </div>

          <div className="text-center mt-4">
            <button
              type="submit"
              className="btn btn-primary btn-lg px-4"
              id="show_visualizations"
              aria-label="Show visualizations"
            >
              <i className="fas fa-chart-bar me-2" aria-hidden="true"></i>
              Show Visualizations
            </button>
          </div>
        </form>
      </section>

      <section aria-live="polite" aria-atomic="true" className="mt-3">
        {errorMessage && (
          <div id="error_message" className="alert alert-danger text-center" role="alert">
            {errorMessage}
          </div>
        )}
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
          {visualizationResults.map((gif, idx) => (
            <div key={idx} className="gif-wrapper">
              <img src={gif} alt={`Animation ${idx + 1}`} className="img-fluid" />
            </div>
          ))}
        </div>
      </section>
    </main>
  );
};

export default VisualizationsDashboard;
