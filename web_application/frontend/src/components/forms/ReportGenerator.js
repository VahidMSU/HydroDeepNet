import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faFileAlt,
  faChevronDown,
  faMapMarkerAlt,
  faCogs,
  faCalendarAlt,
  faCloudSun,
  faMapPin,
  faSpinner,
  faCheckSquare,
  faSquare,
} from '@fortawesome/free-solid-svg-icons';

import {
  QuerySidebarHeader,
  QuerySidebarContent,
  FormGroup,
  InputField,
  SubmitButton,
  InfoCard,
  CoordinatesDisplay,
  CheckboxGroup,
  CheckboxLabel,
} from '../../styles/HydroGeoDataset.tsx';

const ReportGeneratorForm = ({
  reportFormData,
  handleChange,
  handleCheckboxChange,
  handleSubmit,
  isLoading,
  formData,
}) => {
  const hasSelectedArea = Boolean(
    formData.min_latitude &&
      formData.max_latitude &&
      formData.min_longitude &&
      formData.max_longitude,
  );

  const reportTypes = [
    { value: 'all', label: 'All Reports (Comprehensive)' },
    { value: 'prism', label: 'PRISM Climate Report' },
    { value: 'modis', label: 'MODIS Satellite Data Report' },
    { value: 'cdl', label: 'Cropland Data Layer (CDL) Report' },
    { value: 'groundwater', label: 'Groundwater Report' },
    { value: 'gov_units', label: 'Governmental Units Report' },
    { value: 'climate_change', label: 'Climate Change Projections Report' },
    { value: 'nsrdb', label: 'NSRDB Solar Data Report' },
  ];

  return (
    <>
      <QuerySidebarHeader>
        <h2>
          <FontAwesomeIcon icon={faFileAlt} className="icon" />
          Report Generator
        </h2>
      </QuerySidebarHeader>

      <QuerySidebarContent>
        <InfoCard>
          <h3>
            <FontAwesomeIcon icon={faMapMarkerAlt} className="icon" />
            Selected Area
          </h3>
          <p>Generate comprehensive environmental and climate reports for your selected area.</p>
        </InfoCard>

        <form onSubmit={handleSubmit}>
          {/* Report Types */}
          <FormGroup>
            <label>
              <FontAwesomeIcon icon={faFileAlt} className="icon" />
              Report Types
            </label>
            <CheckboxGroup>
              {reportTypes.map((type) => (
                <CheckboxLabel key={type.value}>
                  <input
                    type="checkbox"
                    name={type.value}
                    checked={reportFormData.report_types.includes(type.value)}
                    onChange={handleCheckboxChange}
                  />
                  <FontAwesomeIcon
                    icon={
                      reportFormData.report_types.includes(type.value) ? faCheckSquare : faSquare
                    }
                    className="checkbox-icon"
                  />
                  {type.label}
                </CheckboxLabel>
              ))}
            </CheckboxGroup>
          </FormGroup>

          {/* Time Range */}
          <FormGroup>
            <label>
              <FontAwesomeIcon icon={faCalendarAlt} className="icon" />
              Historical Data Years
            </label>
            <div className="range-inputs">
              <InputField>
                <input
                  type="number"
                  name="start_year"
                  value={reportFormData.start_year}
                  onChange={handleChange}
                  placeholder="Start Year"
                  min="1980"
                  max="2020"
                />
              </InputField>
              <span className="range-separator">to</span>
              <InputField>
                <input
                  type="number"
                  name="end_year"
                  value={reportFormData.end_year}
                  onChange={handleChange}
                  placeholder="End Year"
                  min="1980"
                  max="2020"
                />
              </InputField>
            </div>
          </FormGroup>

          {/* Climate Change Settings */}
          {(reportFormData.report_types.includes('all') ||
            reportFormData.report_types.includes('climate_change')) && (
            <>
              <FormGroup>
                <label>
                  <FontAwesomeIcon icon={faCloudSun} className="icon" />
                  Climate Change Settings
                </label>

                <div className="settings-group">
                  <h4>Historical Period</h4>
                  <div className="range-inputs">
                    <InputField>
                      <input
                        type="number"
                        name="hist_start_year"
                        value={reportFormData.hist_start_year}
                        onChange={handleChange}
                        placeholder="Start Year"
                        min="1980"
                        max="2014"
                      />
                    </InputField>
                    <span className="range-separator">to</span>
                    <InputField>
                      <input
                        type="number"
                        name="hist_end_year"
                        value={reportFormData.hist_end_year}
                        onChange={handleChange}
                        placeholder="End Year"
                        min="1980"
                        max="2014"
                      />
                    </InputField>
                  </div>

                  <h4>Future Period</h4>
                  <div className="range-inputs">
                    <InputField>
                      <input
                        type="number"
                        name="fut_start_year"
                        value={reportFormData.fut_start_year}
                        onChange={handleChange}
                        placeholder="Start Year"
                        min="2015"
                        max="2099"
                      />
                    </InputField>
                    <span className="range-separator">to</span>
                    <InputField>
                      <input
                        type="number"
                        name="fut_end_year"
                        value={reportFormData.fut_end_year}
                        onChange={handleChange}
                        placeholder="End Year"
                        min="2015"
                        max="2100"
                      />
                    </InputField>
                  </div>

                  <h4>Climate Model</h4>
                  <InputField>
                    <select name="cc_model" value={reportFormData.cc_model} onChange={handleChange}>
                      <option value="ACCESS-CM2">ACCESS-CM2</option>
                      <option value="MPI-ESM1-2-HR">MPI-ESM1-2-HR</option>
                      <option value="NorESM2-MM">NorESM2-MM</option>
                      <option value="CNRM-CM6-1">CNRM-CM6-1</option>
                      <option value="CanESM5">CanESM5</option>
                    </select>
                    <FontAwesomeIcon icon={faChevronDown} className="select-arrow" />
                  </InputField>

                  <h4>Scenario</h4>
                  <InputField>
                    <select
                      name="cc_scenario"
                      value={reportFormData.cc_scenario}
                      onChange={handleChange}
                    >
                      <option value="ssp245">SSP245 (Middle of the Road)</option>
                      <option value="ssp370">SSP370 (Regional Rivalry)</option>
                      <option value="ssp585">SSP585 (Fossil-fueled Development)</option>
                    </select>
                    <FontAwesomeIcon icon={faChevronDown} className="select-arrow" />
                  </InputField>

                  <h4>Ensemble</h4>
                  <InputField>
                    <select
                      name="cc_ensemble"
                      value={reportFormData.cc_ensemble}
                      onChange={handleChange}
                    >
                      <option value="r1i1p1f1">r1i1p1f1</option>
                      <option value="r2i1p1f1">r2i1p1f1</option>
                      <option value="r3i1p1f1">r3i1p1f1</option>
                    </select>
                    <FontAwesomeIcon icon={faChevronDown} className="select-arrow" />
                  </InputField>
                </div>
              </FormGroup>
            </>
          )}

          {/* Data Resolution */}
          <FormGroup>
            <label>
              <FontAwesomeIcon icon={faCogs} className="icon" />
              Data Settings
            </label>

            <h4>Resolution (meters)</h4>
            <InputField>
              <select name="resolution" value={reportFormData.resolution} onChange={handleChange}>
                <option value="250">250m (Standard)</option>
                <option value="100">100m (High)</option>
                <option value="500">500m (Low)</option>
              </select>
              <FontAwesomeIcon icon={faChevronDown} className="select-arrow" />
            </InputField>

            <h4>Temporal Aggregation</h4>
            <InputField>
              <select name="aggregation" value={reportFormData.aggregation} onChange={handleChange}>
                <option value="monthly">Monthly</option>
                <option value="seasonal">Seasonal</option>
                <option value="annual">Annual</option>
                <option value="daily">Daily (Large data)</option>
              </select>
              <FontAwesomeIcon icon={faChevronDown} className="select-arrow" />
            </InputField>

            <div style={{ marginTop: '15px' }}>
              <CheckboxLabel>
                <input
                  type="checkbox"
                  name="use_synthetic_data"
                  checked={reportFormData.use_synthetic_data}
                  onChange={(e) =>
                    handleChange({
                      target: {
                        name: 'use_synthetic_data',
                        value: e.target.checked,
                      },
                    })
                  }
                />
                <FontAwesomeIcon
                  icon={reportFormData.use_synthetic_data ? faCheckSquare : faSquare}
                  className="checkbox-icon"
                />
                Use synthetic data if actual data is unavailable
              </CheckboxLabel>
            </div>
          </FormGroup>

          {/* Selected Bounds Display */}
          <FormGroup>
            <label>
              <FontAwesomeIcon icon={faMapPin} className="icon" />
              Area of Interest
            </label>
            <CoordinatesDisplay>
              <div className="title">
                <FontAwesomeIcon icon={faMapMarkerAlt} className="icon" />
                {hasSelectedArea ? 'Area Selected' : 'No Area Selected'}
              </div>
              {hasSelectedArea ? (
                <div className="value">
                  Lat: {formData.min_latitude} to {formData.max_latitude}
                  <br />
                  Lon: {formData.min_longitude} to {formData.max_longitude}
                </div>
              ) : (
                <div className="value warning">
                  Please draw a polygon on the map to define your area of interest.
                </div>
              )}
            </CoordinatesDisplay>
          </FormGroup>

          {/* Submit Button */}
          <SubmitButton type="submit" disabled={!hasSelectedArea || isLoading}>
            {isLoading ? (
              <>
                <FontAwesomeIcon icon={faSpinner} className="icon fa-spin" />
                Generating Reports...
              </>
            ) : (
              <>
                <FontAwesomeIcon icon={faFileAlt} className="icon" />
                Generate Reports
              </>
            )}
          </SubmitButton>
        </form>
      </QuerySidebarContent>
    </>
  );
};

export default ReportGeneratorForm;
