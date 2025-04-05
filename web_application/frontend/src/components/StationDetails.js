import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faMapMarkedAlt,
  faChevronUp,
  faChevronDown,
  faIdCard,
  faLayerGroup,
  faCheckCircle,
  faWater,
  faRulerHorizontal,
  faRulerVertical,
  faStream,
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

  // Format values for display
  const formatValue = (key, value) => {
    if (value === null || value === undefined) return 'N/A';

    switch (key) {
      case 'huc_cd':
        if (typeof value === 'string' || typeof value === 'number') {
          // Ensure it's a string, then pad with leading zeros if needed
          const hucStr = String(value);
          // Truncate or pad to exactly 8 digits
          return hucStr.padStart(8, '0').slice(0, 8);
        }
        return value;
      case 'DrainageArea':
        return `${Number(value).toLocaleString()} km²`;
      case 'StreamflowGapPercent':
        return `${Number(value).toFixed(1)}%`;
      case 'Latitude':
      case 'Longitude':
        return typeof value === 'number' ? `${value.toFixed(4)}°` : value;
      default:
        return value;
    }
  };

  console.log('Station Data in StationDetails:', stationData);

  // Only these fields should be excluded from display
  const excludedFields = [
    'geometries',
    'streams_geometries',
    'lakes_geometries',
    'HUC12 ids of the watershed',
  ];

  // Map keys to icons and beautified labels
  const fieldMappings = {
    SiteNumber: { icon: faIdCard, label: 'Site Number' },
    SiteName: { icon: faMapMarkedAlt, label: 'Station Name' },
    huc_cd: { icon: faLayerGroup, label: 'HUC Code (8-digit)' },
    DrainageArea: { icon: faWater, label: 'Watershed Area' },
    Latitude: { icon: faRulerVertical, label: 'Latitude' },
    Longitude: { icon: faRulerHorizontal, label: 'Longitude' },
    StreamflowGapPercent: { icon: faStream, label: 'Streamflow Gap' },
    ExpectedRecords: { icon: faStream, label: 'Expected Records' },
    Status: { icon: faCheckCircle, label: 'Station Status' },
    USGSFunding: { icon: faCheckCircle, label: 'USGS Funding' },
    'Num HUC12 subbasins': { icon: faLayerGroup, label: 'Number of Subbasins' },
    StationHUC12: { icon: faLayerGroup, label: 'Station HUC12' },
    'HUC12 id of the station': { icon: faLayerGroup, label: 'Station HUC12' },
    site_no: { icon: faIdCard, label: 'USGS Site Number' },
  };

  return (
    <StationDetailsContainer style={{ backgroundColor: '#222222' }}>
      <StationName onClick={toggleExpand} style={{ cursor: 'pointer' }}>
        <StationIcon>
          <FontAwesomeIcon icon={faMapMarkedAlt} />
        </StationIcon>
        {stationData.SiteName &&
        stationData.SiteName.trim() !== '' &&
        stationData.SiteName.trim() !== '---'
          ? stationData.SiteName
          : stationData.SiteNumber}
        <StationToggleIcon>
          <FontAwesomeIcon icon={isExpanded ? faChevronUp : faChevronDown} />
        </StationToggleIcon>
      </StationName>

      {isExpanded && (
        <StationInfoContainer>
          {/* Display all fields with proper formatting */}
          {Object.entries(stationData).map(([key, value]) => {
            // Skip excluded fields or fields that are objects
            if (excludedFields.includes(key) || key.startsWith('_') || typeof value === 'object') {
              return null;
            }

            const mapping = fieldMappings[key] || {
              icon: faIdCard,
              label: key.replace(/([A-Z])/g, ' $1').trim(),
            };

            return (
              <InfoItem key={key}>
                <InfoIcon>
                  <FontAwesomeIcon icon={mapping.icon} />
                </InfoIcon>
                <InfoContent>
                  <InfoLabel>{mapping.label}</InfoLabel>
                  <InfoValue>{formatValue(key, value)}</InfoValue>
                </InfoContent>
              </InfoItem>
            );
          })}
        </StationInfoContainer>
      )}
    </StationDetailsContainer>
  );
});

export default StationDetails;
