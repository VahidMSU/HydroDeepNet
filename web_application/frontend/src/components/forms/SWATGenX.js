import React from 'react';
import SearchForm from '../SearchForm';
import StationDetails from '../StationDetails';
import { CardBody } from '../../styles/Layout.tsx';
import { StyledInput, StyledButton, Section, InfoBox, StrongText } from '../../styles/SWATGenX.tsx';

const ModelSettingsForm = ({
  stationList,
  stationInput,
  setStationInput,
  stationData,
  setStationData,
  lsResolution,
  setLsResolution,
  demResolution,
  setDemResolution,
  calibrationFlag,
  setCalibrationFlag,
  sensitivityFlag,
  setSensitivityFlag,
  validationFlag,
  setValidationFlag,
  handleSubmit,
  loading,
  setLoading,
}) => {
  return (
    <CardBody>
      <SearchForm setStationData={setStationData} setLoading={setLoading} />

      <Section>
        <label>USGS Station Number:</label>
        <StyledInput
          type="text"
          value={stationInput}
          onChange={(e) => setStationInput(e.target.value)}
          placeholder="Enter station number"
          list="station_list"
        />
        <datalist id="station_list">
          {stationList.map((station, index) => (
            <option key={index} value={station} />
          ))}
        </datalist>
      </Section>

      {stationData && <StationDetails stationData={stationData} />}

      <Section>
        <label>Landuse/Soil Resolution:</label>
        <StyledInput
          type="text"
          value={lsResolution}
          onChange={(e) => setLsResolution(e.target.value)}
        />
      </Section>

      <Section>
        <label>DEM Resolution:</label>
        <StyledInput
          type="text"
          value={demResolution}
          onChange={(e) => setDemResolution(e.target.value)}
        />
      </Section>

      <InfoBox>
        <Section>
          <label>
            <StyledInput
              type="checkbox"
              checked={calibrationFlag}
              onChange={(e) => setCalibrationFlag(e.target.checked)}
            />
            <StrongText>Calibration</StrongText>
          </label>
        </Section>

        <Section>
          <label>
            <StyledInput
              type="checkbox"
              checked={sensitivityFlag}
              onChange={(e) => setSensitivityFlag(e.target.checked)}
            />
            <StrongText>Sensitivity Analysis</StrongText>
          </label>
        </Section>

        <Section>
          <label>
            <StyledInput
              type="checkbox"
              checked={validationFlag}
              onChange={(e) => setValidationFlag(e.target.checked)}
            />
            <StrongText>Validation</StrongText>
          </label>
        </Section>
      </InfoBox>

      <StyledButton
        onClick={handleSubmit}
        disabled={loading}
        style={{ marginTop: '1rem', width: '100%' }}
      >
        {loading ? 'Processing...' : 'Run Model'}
      </StyledButton>
    </CardBody>
  );
};

export default ModelSettingsForm;
