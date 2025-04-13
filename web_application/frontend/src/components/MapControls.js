import React, { memo, useCallback } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faSearch,
  faSearchMinus,
  faSearchPlus,
  faTrash,
  faLocationArrow,
  faDrawPolygon,
  faSquare,
  faExpand,
  faMapMarkerAlt
} from '@fortawesome/free-solid-svg-icons';
import styled from '@emotion/styled';
import colors from '../styles/colors.tsx';

const ControlsContainer = styled.div`
  position: absolute;
  top: 15px;
  right: 15px;
  z-index: 100;
  display: flex;
  flex-direction: column;
  gap: 10px;
  background-color: #2b2b2c;
  
  @media (max-width: 768px) {
    top: auto;
    bottom: 10px;
    flex-direction: row;
    flex-wrap: wrap;
  }
`;

const ControlGroup = styled.div`
  display: flex;
  flex-direction: column;
  background-color: #2b2b2c;
  border: 1px solid #444;
  border-radius: 4px;
  overflow: hidden;
`;

const ControlButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background-color: #2b2b2c;
  color: #fff;
  border: none;
  border-bottom: 1px solid #444;
  cursor: pointer;
  position: relative;
  padding: 0;
  transition: background-color 0.2s;

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    background-color: #444;
  }

  &.active {
    background-color: #ff8500;
    color: #fff;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background-color: #2b2b2c;
  }

  .tooltip {
    position: absolute;
    left: -110px;
    background-color: #2b2b2c;
    color: #fff;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 12px;
    white-space: nowrap;
    visibility: hidden;
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
    border: 1px solid #444;
  }

  &:hover .tooltip {
    visibility: visible;
    opacity: 1;
  }
`;

const DrawToolsContainer = styled.div`
  margin-top: 5px;
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding-top: 5px;
  border-top: 1px solid ${colors.border};
  
  @media (max-width: 768px) {
    margin-top: 0;
    padding-top: 0;
    border-top: none;
    padding-left: 5px;
    margin-left: 5px;
    border-left: 1px solid ${colors.border};
    flex-direction: row;
  }
`;

/**
 * MapControls component provides UI controls for map interaction
 */
const MapControls = memo(({ 
  mapRef, 
  onDrawPolygon, 
  onDrawRectangle, 
  onClearGraphics, 
  onLocateMe,
  onZoomIn,
  onZoomOut,
  onExtent,
  activeDrawTool = null,
  disabled = false
}) => {
  // Handlers
  const handleZoomIn = useCallback(() => {
    if (disabled) return;
    if (onZoomIn) {
      onZoomIn();
    } else if (mapRef?.current) {
      const view = mapRef.current.getView();
      if (view) {
        view.zoomIn();
      }
    }
  }, [mapRef, onZoomIn, disabled]);

  const handleZoomOut = useCallback(() => {
    if (disabled) return;
    if (onZoomOut) {
      onZoomOut();
    } else if (mapRef?.current) {
      const view = mapRef.current.getView();
      if (view) {
        view.zoomOut();
      }
    }
  }, [mapRef, onZoomOut, disabled]);

  const handleClear = useCallback(() => {
    if (disabled) return;
    onClearGraphics?.();
  }, [onClearGraphics, disabled]);

  const handleLocate = useCallback(() => {
    if (disabled) return;
    onLocateMe?.();
  }, [onLocateMe, disabled]);

  const handleDrawPolygon = useCallback(() => {
    if (disabled) return;
    onDrawPolygon?.();
  }, [onDrawPolygon, disabled]);

  const handleDrawRectangle = useCallback(() => {
    if (disabled) return;
    onDrawRectangle?.();
  }, [onDrawRectangle, disabled]);

  const handleExtent = useCallback(() => {
    if (disabled) return;
    onExtent?.();
  }, [onExtent, disabled]);

  return (
    <ControlsContainer>
      <ControlButton 
        onClick={handleZoomIn} 
        title="Zoom In"
        className={disabled ? 'disabled' : ''}
      >
        <FontAwesomeIcon icon={faSearchPlus} />
      </ControlButton>
      
      <ControlButton 
        onClick={handleZoomOut} 
        title="Zoom Out"
        className={disabled ? 'disabled' : ''}
      >
        <FontAwesomeIcon icon={faSearchMinus} />
      </ControlButton>
      
      <ControlButton 
        onClick={handleLocate} 
        title="My Location"
        className={disabled ? 'disabled' : ''}
      >
        <FontAwesomeIcon icon={faLocationArrow} />
      </ControlButton>
      
      <ControlButton 
        onClick={handleExtent} 
        title="Zoom to Full Extent"
        className={disabled ? 'disabled' : ''}
      >
        <FontAwesomeIcon icon={faExpand} />
      </ControlButton>
      
      <ControlButton 
        onClick={handleClear} 
        title="Clear Graphics"
        className={disabled ? 'disabled' : ''}
      >
        <FontAwesomeIcon icon={faTrash} />
      </ControlButton>
      
      <DrawToolsContainer>
        <ControlButton 
          onClick={handleDrawPolygon} 
          title="Draw Polygon"
          className={`${disabled ? 'disabled' : ''} ${activeDrawTool === 'polygon' ? 'active' : ''}`}
        >
          <FontAwesomeIcon icon={faDrawPolygon} />
        </ControlButton>
        
        <ControlButton 
          onClick={handleDrawRectangle} 
          title="Draw Rectangle"
          className={`${disabled ? 'disabled' : ''} ${activeDrawTool === 'rectangle' ? 'active' : ''}`}
        >
          <FontAwesomeIcon icon={faSquare} />
        </ControlButton>
        
        <ControlButton 
          onClick={handleClear} 
          title="Place Point"
          className={`${disabled ? 'disabled' : ''} ${activeDrawTool === 'point' ? 'active' : ''}`}
        >
          <FontAwesomeIcon icon={faMapMarkerAlt} />
        </ControlButton>
      </DrawToolsContainer>
    </ControlsContainer>
  );
});

MapControls.displayName = 'MapControls';

export default MapControls;
