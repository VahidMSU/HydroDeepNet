import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faSearch,
  faTimes,
  faExclamationTriangle,
  faMousePointer,
} from '@fortawesome/free-solid-svg-icons';
import {
  SearchForm as StyledSearchForm,
  SearchInputGroup,
  SearchInputWrapper,
  FormInput,
  SearchButton,
  SearchResults,
  SearchResultItem,
  FeedbackMessage,
  FeedbackIcon,
} from '../styles/SWATGenX.tsx';

function SearchForm({
  setStationData,
  setLoading,
  mapSelections = [],
  handleStationSelect = null, // Receive the handler from parent
}) {
  const [searchInput, setSearchInput] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [mapMode, setMapMode] = useState(true); // True if map mode is enabled
  const [searchError, setSearchError] = useState('');
  const [siteNumberInput, setSiteNumberInput] = useState(''); // New state for site number input

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
    setSearchError('');

    try {
      const response = await fetch(`/api/search_site?search_term=${searchInput}`);
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();

      if (data.error) {
        setSearchResults([]);
        setSearchError('No stations found matching your search term');
      } else {
        setSearchResults(data);
        if (data.length === 0) {
          setSearchError('No stations found matching your search term');
        }
      }
    } catch (error) {
      console.error('Error fetching search results:', error);
      setSearchError(`Error searching for stations: ${error.message}`);
      setSearchResults([]);
    }
    setLoading(false);
  };

  // Handle Enter key press in search input
  const handleSearchKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
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
      setSearchError(`Error fetching station details: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle direct site number input
  const handleSiteNumberSubmit = () => {
    if (siteNumberInput.trim()) {
      onStationSelect(siteNumberInput.trim());
    }
  };

  return (
    <StyledSearchForm>
      <SearchInputGroup>
        <label>Search or Select Station:</label>

        <SearchInputWrapper>
          <FormInput
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            placeholder="Enter station name or number"
          />
          <SearchButton onClick={handleSearch}>
            <FontAwesomeIcon icon={faSearch} />
          </SearchButton>
        </SearchInputWrapper>

        {/* New field for direct site number input */}
        <SearchInputWrapper style={{ marginTop: '10px' }}>
          <FormInput
            type="text"
            value={siteNumberInput}
            onChange={(e) => setSiteNumberInput(e.target.value)}
            placeholder="Enter site number directly"
          />
          <SearchButton onClick={handleSiteNumberSubmit}>
            <FontAwesomeIcon icon={faSearch} />
          </SearchButton>
        </SearchInputWrapper>
      </SearchInputGroup>

      {searchError && (
        <FeedbackMessage type="error" style={{ margin: '10px 0' }}>
          <FeedbackIcon>
            <FontAwesomeIcon icon={faExclamationTriangle} />
          </FeedbackIcon>
          <span>{searchError}</span>
        </FeedbackMessage>
      )}

      {searchResults.length > 0 && (
        <SearchResults>
          <div style={{ marginBottom: '8px', fontSize: '14px' }}>
            {mapMode
              ? `${searchResults.length} station${searchResults.length > 1 ? 's' : ''} selected. Click one to view details.`
              : 'Search results:'}
          </div>
          {searchResults.map((site) => (
            <SearchResultItem
              key={site.SiteNumber}
              onClick={() => onStationSelect(site.SiteNumber)}
            >
              <strong>{site.SiteName}</strong>
              <span>(ID: {site.SiteNumber})</span>
            </SearchResultItem>
          ))}
        </SearchResults>
      )}

      {mapMode && !searchResults.length && !searchError && (
        <div style={{ margin: '15px 0', fontSize: '14px', color: '#666' }}>
          <p>Select a station by:</p>
          <ul style={{ paddingLeft: '20px', marginTop: '5px' }}>
            <li>Clicking directly on the map</li>
            <li>Using the search box above</li>
          </ul>
        </div>
      )}
    </StyledSearchForm>
  );
}

export default SearchForm;
