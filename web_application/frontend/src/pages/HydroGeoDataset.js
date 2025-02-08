// pages/HydroGeoDataset.js
import React, { useState } from 'react';
import MapComponent from '../components/MapComponent';
import FormComponent from '../components/FormComponent';
import '../styles/HydroGeoDataset.tsx'; // Ensure the path is correct
import '../styles/Layout.tsx'; // Ensure the path is correct

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
  const [data, setData] = useState(null);

  return (
    <div className="container">
      {/* Map Section */}
      <MapComponent setFormData={setFormData} />

      {/* Form Section */}
      <FormComponent formData={formData} setFormData={setFormData} data={data} setData={setData} />
    </div>
  );
};

export default HydroGeoDataset;
