import React from 'react';
import { Box } from '@mui/material';

import {
  FormContainer,
  FormGrid,
  FormGroup,
  FormLabel,
  FormSelect,
  FormInput,
  HelpText,
  SubmitButton,
  ErrorMessage,
} from '../../styles/Visualization.tsx';

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
    <FormContainer onSubmit={handleSubmit} id="visualization_form">
      <Box display="flex" justifyContent="center">
        <FormGrid>
          <FormGroup>
            <FormLabel htmlFor="NAME">Watershed Name</FormLabel>
            <FormSelect
              name="NAME"
              id="NAME"
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
            </FormSelect>
          </FormGroup>

          <FormGroup className="ensemble-group">
            <FormLabel htmlFor="ver">Ensemble</FormLabel>
            <FormInput
              type="text"
              name="ver"
              id="ver"
              placeholder="Enter Ensemble Version"
              required
              value={ensemble}
              onChange={(e) => setEnsemble(e.target.value)}
            />
          </FormGroup>

          <FormGroup>
            <FormLabel htmlFor="variable">Variable Name</FormLabel>
            <FormSelect
              name="variable"
              id="variable"
              multiple
              value={selectedVariables}
              onChange={handleVariableChange}
            >
              {availableVariables.map((variable, idx) => (
                <option key={idx} value={variable}>
                  {variable}
                </option>
              ))}
            </FormSelect>
            <HelpText id="variable-help">Hold Ctrl/Cmd to select multiple variables.</HelpText>
          </FormGroup>
        </FormGrid>
      </Box>

      <Box display="flex" justifyContent="center" mt={4}>
        <SubmitButton type="submit" id="show_visualizations" aria-label="Show visualizations">
          <i className="fas fa-chart-bar me-2" aria-hidden="true"></i>
          Show Visualizations
        </SubmitButton>
      </Box>

      {errorMessage && (
        <ErrorMessage id="error_message" role="alert">
          {errorMessage}
        </ErrorMessage>
      )}
    </FormContainer>
  );
};

export default VisualizationForm;
