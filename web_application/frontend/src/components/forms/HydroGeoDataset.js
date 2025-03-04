import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faLayerGroup,
  faChevronDown,
  faMapMarkerAlt,
  faFilter,
  faDatabase,
  faSpinner,
} from '@fortawesome/free-solid-svg-icons';

import {
  QuerySidebarHeader,
  QuerySidebarContent,
  FormGroup,
  InputField,
  SubmitButton,
  InfoCard,
  CoordinatesDisplay,
} from '../../styles/HydroGeoDataset.tsx';

const HydroGeoDatasetForm = ({
  formData,
  handleChange,
  handleSubmit,
  availableVariables,
  availableSubvariables,
  isLoading,
}) => {
  // Check if area is selected by verifying coordinate bounds
  const hasSelectedArea = Boolean(
    formData.min_latitude &&
      formData.max_latitude &&
      formData.min_longitude &&
      formData.max_longitude,
  );

  // Debug logging to understand the form state
  console.log('Form data in component:', {
    hasArea: hasSelectedArea,
    bounds: {
      minLat: formData.min_latitude,
      maxLat: formData.max_latitude,
      minLon: formData.min_longitude,
      maxLon: formData.max_longitude,
    },
    variable: formData.variable,
    subvariable: formData.subvariable,
  });

  return (
    <>
      <QuerySidebarHeader>
        <h2>
          <FontAwesomeIcon icon={faFilter} className="icon" />
          Query Parameters
        </h2>
      </QuerySidebarHeader>

      <QuerySidebarContent>
        <InfoCard>
          <h3>
            <FontAwesomeIcon icon={faMapMarkerAlt} className="icon" />
            Select Area
          </h3>
          <p>Use the map to draw a polygon around your area of interest.</p>
        </InfoCard>

        <form onSubmit={handleSubmit}>
          {/* Variable Select */}
          <FormGroup>
            <label>
              <FontAwesomeIcon icon={faLayerGroup} className="icon" />
              Variable
            </label>
            <InputField>
              <select name="variable" value={formData.variable} onChange={handleChange} required>
                <option value="">Select a variable</option>
                {availableVariables.map((variable) => (
                  <option key={variable} value={variable}>
                    {variable}
                  </option>
                ))}
              </select>
              <FontAwesomeIcon icon={faChevronDown} className="select-arrow" />
            </InputField>
          </FormGroup>

          {/* Subvariable Select */}
          {formData.variable && (
            <FormGroup>
              <label>
                <FontAwesomeIcon icon={faFilter} className="icon" />
                Subvariable
              </label>
              <InputField>
                <select
                  name="subvariable"
                  value={formData.subvariable}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select a subvariable</option>
                  {availableSubvariables.map((subvariable) => (
                    <option key={subvariable} value={subvariable}>
                      {subvariable}
                    </option>
                  ))}
                </select>
                <FontAwesomeIcon icon={faChevronDown} className="select-arrow" />
              </InputField>
            </FormGroup>
          )}

          {/* If bounds are defined, show them */}
          {hasSelectedArea && (
            <FormGroup>
              <label>
                <FontAwesomeIcon icon={faMapMarkerAlt} className="icon" />
                Bounds
              </label>
              <CoordinatesDisplay>
                <div className="title">
                  <FontAwesomeIcon icon={faMapMarkerAlt} className="icon" />
                  Coordinate Bounds
                </div>
                <div className="value">
                  Lat: {formData.min_latitude} to {formData.max_latitude}
                  <br />
                  Lon: {formData.min_longitude} to {formData.max_longitude}
                </div>
              </CoordinatesDisplay>
            </FormGroup>
          )}

          {/* Submit Button */}
          <SubmitButton
            type="submit"
            disabled={!hasSelectedArea || !formData.variable || !formData.subvariable || isLoading}
          >
            {isLoading ? (
              <>
                <FontAwesomeIcon icon={faSpinner} className="icon fa-spin" />
                Processing...
              </>
            ) : (
              <>
                <FontAwesomeIcon icon={faDatabase} className="icon" />
                Fetch Data
              </>
            )}
          </SubmitButton>

          {/* Show explanation text if button is disabled */}
          {(!hasSelectedArea || !formData.variable || !formData.subvariable) && (
            <div style={{ marginTop: '10px', color: '#666', fontSize: '0.9rem' }}>
              {!hasSelectedArea && <div>⚠️ Please draw an area on the map</div>}
              {!formData.variable && <div>⚠️ Please select a variable</div>}
              {formData.variable && !formData.subvariable && (
                <div>⚠️ Please select a subvariable</div>
              )}
            </div>
          )}
        </form>
      </QuerySidebarContent>
    </>
  );
};

export default HydroGeoDatasetForm;
