import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearch, faDrawPolygon, faTimes } from '@fortawesome/free-solid-svg-icons';
import {
  SearchForm as StyledSearchForm,
  SearchInputGroup,
  SearchInputWrapper,
  FormInput,
  SearchButton,
  SearchResults,
  SearchResultItem,
} from '../styles/SWATGenX.tsx';

function SearchForm({
  setStationData,
  setLoading,
  mapSelections = [],
  setDrawingMode = null,
  drawingMode = false,
  handleStationSelect = null, // Receive the handler from parent
}) {
  const [searchInput, setSearchInput] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [mapMode, setMapMode] = useState(!!setDrawingMode); // True if map mode is enabled

  // Update search results when map selections change
  useEffect(() => {
    if (mapSelections.length > 0) {
      setSearchResults(mapSelections);
    }
  }, [mapSelections]);

  // Fetch search results from `/search_site`
  const handleSearch = async () => {
    if (!searchInput.trim()) {
      return;
    }
    setLoading(true);

    try {
      const response = await fetch(`/api/search_site?search_term=${searchInput}`);
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      setSearchResults(data.error ? [] : data);
    } catch (error) {
      console.error('Error fetching search results:', error);
    }
    setLoading(false);
  };

  // Handle station selection - use provided handler if available
  const onStationSelect = (stationNumber) => {
    if (handleStationSelect) {
      handleStationSelect(stationNumber);
    } else {
      handleDefaultStationSelect(stationNumber);
    }
  };

  // Default station selection handler
  const handleDefaultStationSelect = async (stationNumber) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/get_station_characteristics?station=${stationNumber}`);
      const data = await response.json();
      setStationData(data);
    } catch (error) {
      console.error('Error fetching station details:', error);
    } finally {
      setLoading(false);
    }
  };

  // Toggle drawing mode
  const toggleDrawingMode = () => {
    if (setDrawingMode) {
      setDrawingMode(!drawingMode);
    }
  };

  return (
    <StyledSearchForm>
      <SearchInputGroup>
        <label>Search Site Name:</label>
        <SearchInputWrapper>
          <FormInput
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Enter site name"
          />
          <SearchButton onClick={handleSearch}>
            <FontAwesomeIcon icon={faSearch} />
          </SearchButton>
          {setDrawingMode && (
            <SearchButton
              onClick={toggleDrawingMode}
              style={{
                marginLeft: '5px',
                backgroundColor: drawingMode ? '#f14668' : '#3273dc',
              }}
              title={drawingMode ? 'Cancel drawing' : 'Draw polygon to select stations'}
            >
              <FontAwesomeIcon icon={drawingMode ? faTimes : faDrawPolygon} />
            </SearchButton>
          )}
        </SearchInputWrapper>
      </SearchInputGroup>

      {searchResults.length > 0 && (
        <SearchResults>
          <div style={{ marginBottom: '8px', fontSize: '14px' }}>
            {mapMode && drawingMode
              ? 'Drawing mode active. Draw a polygon to select stations.'
              : mapMode
                ? `${searchResults.length} stations selected. Click one to view details.`
                : 'Search results:'}
          </div>
          {searchResults.map((site) => (
            <SearchResultItem
              key={site.SiteNumber}
              onClick={() => onStationSelect(site.SiteNumber)}
            >
              <strong>{site.SiteName}</strong>
              <span>(Number: {site.SiteNumber})</span>
            </SearchResultItem>
          ))}
        </SearchResults>
      )}

      {mapMode && !searchResults.length && !drawingMode && (
        <div style={{ margin: '15px 0', fontSize: '14px', color: '#666' }}>
          Use the draw tool to select stations on the map, or search by name above.
        </div>
      )}
    </StyledSearchForm>
  );
}

export default SearchForm;
