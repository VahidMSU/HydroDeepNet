import React, { useState, useEffect } from 'react';
import MapComponent from '../MapComponent';
import HydroGeoDatasetForm from '../forms/HydroGeoDataset';
import {
  PageLayout,
  Sidebar,
  MapContainer,
  DataDisplay,
  Title,
} from '../../styles/HydroGeoDataset.tsx';

const HydroGeoDataset = () => {
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
    <PageLayout>
      <Sidebar>
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
          <DataDisplay>
            <Title>Results</Title>
            <pre>{JSON.stringify(data, null, 2)}</pre>
          </DataDisplay>
        )}
      </Sidebar>

      <MapContainer>
        <MapComponent setFormData={setFormData} />
      </MapContainer>
    </PageLayout>
  );
};

export default HydroGeoDataset;
