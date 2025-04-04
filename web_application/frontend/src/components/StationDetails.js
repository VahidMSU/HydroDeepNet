import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faLocationDot,
  faWater,
  faRulerVertical,
  faRulerHorizontal,
  faMapMarkedAlt,
  faChevronUp,
  faChevronDown,
} from '@fortawesome/free-solid-svg-icons';
import {
  StationDetailsContainer,
  StationName,
  StationIcon,
  StationInfoContainer,
  InfoItem,
  InfoIcon,
  InfoContent,
  InfoLabel,
  InfoValue,
  StationToggleIcon,
} from '../styles/SWATGenX.tsx';

// Using React.memo to prevent unnecessary re-renders
const StationDetails = React.memo(({ stationData }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!stationData) return null;

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <StationDetailsContainer>
      <StationName onClick={toggleExpand} style={{ cursor: 'pointer' }}>
        <StationIcon>
          <FontAwesomeIcon icon={faMapMarkedAlt} />
        </StationIcon>
        {stationData.SiteName || 'Station Details'}
        <StationToggleIcon>
          <FontAwesomeIcon icon={isExpanded ? faChevronUp : faChevronDown} />
        </StationToggleIcon>
      </StationName>

      {isExpanded && (
        <StationInfoContainer>
          <InfoItem>
            <InfoIcon>
              <FontAwesomeIcon icon={faLocationDot} />
            </InfoIcon>
            <InfoContent>
              <InfoLabel>Site Number</InfoLabel>
              <InfoValue>{stationData.SiteNumber || 'N/A'}</InfoValue>
            </InfoContent>
          </InfoItem>
          <InfoItem>
            <InfoIcon>
              <FontAwesomeIcon icon={faWater} />
            </InfoIcon>
            <InfoContent>
              <InfoLabel>Watershed Area</InfoLabel>
              <InfoValue>
                {stationData.DrainageArea ? `${stationData.DrainageArea} km²` : 'N/A'}
              </InfoValue>
            </InfoContent>
          </InfoItem>
          <InfoItem>
            <InfoIcon>
              <FontAwesomeIcon icon={faRulerVertical} />
            </InfoIcon>
            <InfoContent>
              <InfoLabel>Latitude</InfoLabel>
              <InfoValue>
                {stationData.Latitude ? `${stationData.Latitude.toFixed(4)}°` : 'N/A'}
              </InfoValue>
            </InfoContent>
          </InfoItem>
          <InfoItem>
            <InfoIcon>
              <FontAwesomeIcon icon={faRulerHorizontal} />
            </InfoIcon>
            <InfoContent>
              <InfoLabel>Longitude</InfoLabel>
              <InfoValue>
                {stationData.Longitude ? `${stationData.Longitude.toFixed(4)}°` : 'N/A'}
              </InfoValue>
            </InfoContent>
          </InfoItem>
        </StationInfoContainer>
      )}
    </StationDetailsContainer>
  );
});

export default StationDetails;
