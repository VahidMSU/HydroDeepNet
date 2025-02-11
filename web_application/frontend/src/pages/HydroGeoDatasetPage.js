import React, { useState, useEffect } from 'react';
import HydroGeoDataset from '../components/templates/HydroGeoDataset';
import HydroGeoDatasetForm from '../components/forms/HydroGeoDataset';

const HydroGeoDatasetPage = () => {
  const [formData, setFormData] = useState({
    min_latitude: '',
    max_latitude: '',
    min_longitude: '',
    max_longitude: '',
    variable: '',
    subvariable: '',
    geometry: null,
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

    // Log full formData before sending
    console.log('Submitting data with formData:', JSON.stringify(formData, null, 2));

    // Check if polygon exists and convert it to JSON string
    let payload = { ...formData };
    if (formData.geometry && formData.geometry.rings) {
      payload.polygon_coordinates = JSON.stringify(
        formData.geometry.rings[0].map((coord) => ({
          latitude: parseFloat(coord[1]),
          longitude: parseFloat(coord[0]),
        })),
      );
    }

    console.log('Final payload:', JSON.stringify(payload, null, 2));

    try {
      const response = await fetch('/hydro_geo_dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      setData(data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  return (
    <div style={{ display: 'flex' }}>
      <HydroGeoDatasetForm
        formData={formData}
        handleChange={handleChange}
        handleSubmit={handleSubmit}
        availableVariables={availableVariables}
        availableSubvariables={availableSubvariables}
      />
      <HydroGeoDataset
        formData={formData}
        handleChange={handleChange}
        handleSubmit={handleSubmit}
        availableVariables={availableVariables}
        availableSubvariables={availableSubvariables}
        data={data}
        setData={setData}
      />
    </div>
  );
};

export default HydroGeoDatasetPage;
