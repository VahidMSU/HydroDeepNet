import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faLayerGroup,
  faChevronDown,
  faMapMarkerAlt,
  faSearchLocation,
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
  const hasSelectedArea = Boolean(formData.geometry);

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

          {/* Selected Area Display */}
          <FormGroup>
            <label>
              <FontAwesomeIcon icon={faSearchLocation} className="icon" />
              Selected Area
            </label>
            <CoordinatesDisplay>
              <div className="title">
                <FontAwesomeIcon icon={faMapMarkerAlt} className="icon" />
                {hasSelectedArea ? 'Area Selected' : 'No Area Selected'}
              </div>
              <div className="value">
                {hasSelectedArea
                  ? `Polygon with ${formData.geometry.rings[0].length} vertices`
                  : 'Use the map to draw a polygon'}
              </div>
            </CoordinatesDisplay>
          </FormGroup>

          {/* If bounds are defined, show them */}
          {formData.min_latitude && (
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
        </form>
      </QuerySidebarContent>
    </>
  );
};

export default HydroGeoDatasetForm;
