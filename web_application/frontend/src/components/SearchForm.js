import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearch } from '@fortawesome/free-solid-svg-icons';
import {
  SearchForm as StyledSearchForm,
  SearchInputGroup,
  SearchInputWrapper,
  FormInput,
  SearchButton,
  SearchResults,
  SearchResultItem,
} from '../styles/SWATGenX.tsx';

function SearchForm({ setStationData, setLoading }) {
  const [searchInput, setSearchInput] = useState('');
  const [searchResults, setSearchResults] = useState([]);

  // ✅ Fetch search results from `/search_site`
  const handleSearch = async () => {
    if (!searchInput.trim()) {
      return;
    }
    setLoading(true);

    try {
      const response = await fetch(`/search_site?search_term=${searchInput}`);
      const data = await response.json();
      setSearchResults(data.error ? [] : data);
    } catch (error) {
      console.error('Error fetching search results:', error);
    }
    setLoading(false);
  };

  // ✅ Fetch selected station characteristics
  const handleStationSelect = async (stationNumber) => {
    setLoading(true);
    try {
      const response = await fetch(`/get_station_characteristics?station=${stationNumber}`);
      const data = await response.json();
      setStationData(data);
    } catch (error) {
      console.error('Error fetching station details:', error);
    }
    setLoading(false);
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
        </SearchInputWrapper>
      </SearchInputGroup>

      {searchResults.length > 0 && (
        <SearchResults>
          {searchResults.map((site) => (
            <SearchResultItem
              key={site.SiteNumber}
              onClick={() => handleStationSelect(site.SiteNumber)}
            >
              <strong>{site.SiteName}</strong>
              <span>(Number: {site.SiteNumber})</span>
            </SearchResultItem>
          ))}
        </SearchResults>
      )}
    </StyledSearchForm>
  );
}

export default SearchForm;
