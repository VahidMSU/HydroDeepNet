import React, { useState, useEffect } from 'react';
import MapComponent from '../MapComponent'; // Ensure the path is correct
import HydroGeoDatasetForm from '../forms/HydroGeoDataset'; // Ensure the path is correct

const HydroGeoDatasetTemplate = () => {
  const [formData, setFormData] = useState({
    latitude: '',
    longitude: '',
    min_latitude: '',
    max_latitude: '',
    min_longitude: '',
    max_longitude: '',
    variable: '',
    subvariable: '',
  });
  const [availableVariables, setAvailableVariables] = useState([]);
  const [availableSubvariables, setAvailableSubvariables] = useState([]);
  const [data, setData] = useState(null);

  // Fetch available variables on component mount
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await fetch('/hydro_geo_dataset');
        const data = await response.json();
        setAvailableVariables(data.variables);
      } catch (error) {
        console.error('Error fetching options:', error);
      }
    };

    fetchOptions();
  }, []);

  // Fetch subvariables when variable changes
  useEffect(() => {
    const fetchSubvariables = async () => {
      try {
        const response = await fetch(`/hydro_geo_dataset?variable=${formData.variable}`);
        const data = await response.json();
        setAvailableSubvariables(data.subvariables);
      } catch (error) {
        console.error('Error fetching subvariables:', error);
      }
    };

    if (formData.variable) {
      fetchSubvariables();
    }
  }, [formData.variable]);

  const handleChange = ({ target: { name, value } }) => {
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('/hydro_geo_dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setData(data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  return (
    <div className="container">
      {/* Map Section */}
      <MapComponent setFormData={setFormData} />

      {/* Form Section */}
      <HydroGeoDatasetForm
        formData={formData}
        handleChange={handleChange}
        handleSubmit={handleSubmit}
        availableVariables={availableVariables}
        availableSubvariables={availableSubvariables}
      />

      {/* Display Data */}
      {data && (
        <div className="data-display">
          <h3>Data:</h3>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default HydroGeoDatasetTemplate;
