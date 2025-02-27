import React from 'react';
import SearchForm from '../SearchForm';
import StationDetails from '../StationDetails';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faRuler, faLayerGroup, faMapMarkedAlt, faPlay } from '@fortawesome/free-solid-svg-icons';
import {
  ModelSettingsFormContainer,
  FormSection,
  SectionTitle,
  SectionIcon,
  InputGroup,
  InputLabel,
  FormInput,
  CheckboxGroup,
  CheckboxLabel,
  CheckboxInput,
  CheckboxCustom,
  CheckboxText,
  SubmitButton,
  ButtonIcon,
  LoadingSpinner,
} from '../../styles/SWATGenX.tsx';

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
    <ModelSettingsFormContainer>
      <FormSection>
        <SectionTitle>
          <SectionIcon>
            <FontAwesomeIcon icon={faMapMarkedAlt} />
          </SectionIcon>
          Station Selection
        </SectionTitle>

        <SearchForm setStationData={setStationData} setLoading={setLoading} />

        <InputGroup>
          <InputLabel htmlFor="station-number">USGS Station Number:</InputLabel>
          <FormInput
            id="station-number"
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
        </InputGroup>

        {stationData && <StationDetails stationData={stationData} />}
      </FormSection>

      <FormSection>
        <SectionTitle>
          <SectionIcon>
            <FontAwesomeIcon icon={faRuler} />
          </SectionIcon>
          Resolution Settings
        </SectionTitle>

        <InputGroup>
          <InputLabel htmlFor="ls-resolution">Landuse/Soil Resolution:</InputLabel>
          <FormInput
            id="ls-resolution"
            type="text"
            value={lsResolution}
            onChange={(e) => setLsResolution(e.target.value)}
            placeholder="Enter resolution (e.g., 250)"
          />
        </InputGroup>

        <InputGroup>
          <InputLabel htmlFor="dem-resolution">DEM Resolution:</InputLabel>
          <FormInput
            id="dem-resolution"
            type="text"
            value={demResolution}
            onChange={(e) => setDemResolution(e.target.value)}
            placeholder="Enter resolution (e.g., 30)"
          />
        </InputGroup>
      </FormSection>

      <FormSection>
        <SectionTitle>
          <SectionIcon>
            <FontAwesomeIcon icon={faLayerGroup} />
          </SectionIcon>
          Analysis Options
        </SectionTitle>

        <CheckboxGroup>
          <CheckboxLabel>
            <CheckboxInput
              type="checkbox"
              checked={calibrationFlag}
              onChange={(e) => setCalibrationFlag(e.target.checked)}
            />
            <CheckboxCustom checked={calibrationFlag} />
            <CheckboxText>Calibration</CheckboxText>
          </CheckboxLabel>
        </CheckboxGroup>

        <CheckboxGroup>
          <CheckboxLabel>
            <CheckboxInput
              type="checkbox"
              checked={sensitivityFlag}
              onChange={(e) => setSensitivityFlag(e.target.checked)}
            />
            <CheckboxCustom checked={sensitivityFlag} />
            <CheckboxText>Sensitivity Analysis</CheckboxText>
          </CheckboxLabel>
        </CheckboxGroup>

        <CheckboxGroup>
          <CheckboxLabel>
            <CheckboxInput
              type="checkbox"
              checked={validationFlag}
              onChange={(e) => setValidationFlag(e.target.checked)}
            />
            <CheckboxCustom checked={validationFlag} />
            <CheckboxText>Validation</CheckboxText>
          </CheckboxLabel>
        </CheckboxGroup>
      </FormSection>

      <SubmitButton onClick={handleSubmit} isLoading={loading} disabled={loading}>
        {loading ? (
          <>
            <LoadingSpinner />
            Processing...
          </>
        ) : (
          <>
            <ButtonIcon>
              <FontAwesomeIcon icon={faPlay} />
            </ButtonIcon>
            Run Model
          </>
        )}
      </SubmitButton>
    </ModelSettingsFormContainer>
  );
};

export default ModelSettingsForm;
