import React from 'react';
import { Box } from '@mui/material';

const VisualizationForm = ({
  watersheds,
  selectedWatershed,
  setSelectedWatershed,
  ensemble,
  setEnsemble,
  availableVariables,
  selectedVariables,
  setSelectedVariables,
  handleSubmit,
  errorMessage,
}) => {
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
    <form id="visualization_form" className="form-container" onSubmit={handleSubmit}>
      <Box display="flex" justifyContent="center">
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

          <div className="form-group ensemble-group">
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
      </Box>

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

      {errorMessage && (
        <div id="error_message" className="alert alert-danger text-center mt-3" role="alert">
          {errorMessage}
        </div>
      )}
    </form>
  );
};

export default VisualizationForm;
