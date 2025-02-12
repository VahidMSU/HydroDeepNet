import React from 'react';
import {
  StyledForm,
  FormField,
  Label,
  Select,
  Input,
  ReadOnlyInput,
  SubmitButton,
  CoordinatesDisplay,
  CoordinateField,
  CoordinateLabel,
} from '../../styles/HydroGeoDataset.tsx';

const HydroGeoDatasetForm = ({
  formData,
  handleChange,
  handleSubmit,
  availableVariables,
  availableSubvariables,
}) => {
  return (
    <StyledForm onSubmit={handleSubmit}>
      <FormField>
        <Label htmlFor="variable">Variable:</Label>
        <Select id="variable" name="variable" value={formData.variable} onChange={handleChange}>
          <option value="">Select Variable</option>
          {availableVariables.map((variable) => (
            <option key={variable} value={variable}>
              {variable}
            </option>
          ))}
        </Select>
      </FormField>

      <FormField>
        <Label htmlFor="subvariable">Subvariable:</Label>
        <Select
          id="subvariable"
          name="subvariable"
          value={formData.subvariable}
          onChange={handleChange}
        >
          <option value="">Select Subvariable</option>
          {availableSubvariables.map((subvariable) => (
            <option key={subvariable} value={subvariable}>
              {subvariable}
            </option>
          ))}
        </Select>
      </FormField>

      <CoordinatesDisplay>
        <CoordinateField>
          <CoordinateLabel>Selected Coordinates:</CoordinateLabel>
          <ReadOnlyInput
            type="text"
            value={formData.geometry ? 'Area selected on map' : 'No area selected'}
            disabled
          />
        </CoordinateField>

        <CoordinateField>
          <CoordinateLabel>Bounds:</CoordinateLabel>
          <ReadOnlyInput
            type="text"
            value={
              formData.min_latitude
                ? `Lat: ${formData.min_latitude} to ${formData.max_latitude}, Lon: ${formData.min_longitude} to ${formData.max_longitude}`
                : 'Use map to select area'
            }
            disabled
          />
        </CoordinateField>
      </CoordinatesDisplay>

      <SubmitButton type="submit" variant="contained" disabled={!formData.geometry}>
        Fetch Data
      </SubmitButton>
    </StyledForm>
  );
};

export default HydroGeoDatasetForm;
