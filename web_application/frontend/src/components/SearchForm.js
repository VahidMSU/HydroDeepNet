// componenets/SearchForm.js
import React, { useState } from 'react';

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
    <div className="card">
      <div className="card-body">
        {/* Search Form */}
        <div className="form-group">
          <label>Search Site Name:</label>
          <input
            type="text"
            className="form-control"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Enter site name"
          />
          <button className="btn btn-secondary mt-2" onClick={handleSearch}>
            Search
          </button>
        </div>

        {/* Search Results */}
        {searchResults.length > 0 && (
          <ul>
            {searchResults.map((site) => (
              <li
                key={site.SiteNumber}
                style={{ cursor: 'pointer', color: 'blue' }}
                onClick={() => handleStationSelect(site.SiteNumber)}
              >
                <strong>{site.SiteName}</strong> (Number: {site.SiteNumber})
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default SearchForm;
