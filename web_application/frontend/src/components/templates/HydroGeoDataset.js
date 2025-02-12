import React, { useState, useEffect } from 'react';
import MapComponent from '../MapComponent';
import HydroGeoDatasetForm from '../forms/HydroGeoDataset';
import {
  PageLayout,
  Sidebar,
  MapContainer,
  DataDisplay,
  Title,
  InfoSection,
  InfoContent,
  SectionHeader,
  DataResults,
  Collapsible,
  CollapsibleHeader,
  CollapsibleContent,
} from '../../styles/HydroGeoDataset.tsx';
import * as webMercatorUtils from '@arcgis/core/geometry/support/webMercatorUtils';

const HydroGeoDataset = () => {
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
  const [isCollapsed, setIsCollapsed] = useState(true);

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

  const handleGeometryChange = (geom) => {
    if (!geom) {
      console.warn('Received null or undefined geometry.');
      return;
    }
    let convertedGeometry = { ...geom };
    if (geom.rings) {
      convertedGeometry.rings = geom.rings.map((ring) =>
        ring.map((coord) => {
          const geographicPoint = webMercatorUtils.xyToLngLat(coord[0], coord[1]);
          return [geographicPoint[0].toFixed(6), geographicPoint[1].toFixed(6)];
        }),
      );
    }
    setFormData((prev) => ({ ...prev, geometry: convertedGeometry }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('Submitting data with formData:', JSON.stringify(formData, null, 2));

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
    <PageLayout>
      <Sidebar>
        <SectionHeader>HydroGeoDataset</SectionHeader>

        <Collapsible>
          <CollapsibleHeader onClick={() => setIsCollapsed(!isCollapsed)}>
            {isCollapsed ? '▶' : '▼'} Dataset Overview
          </CollapsibleHeader>
          {!isCollapsed && (
            <CollapsibleContent>
              <InfoSection>
                <InfoContent>
                  <p>
                    <strong>HydroGeoDataset</strong> is an integrated geospatial data platform
                    providing access to high-resolution hydrological, environmental, and climate
                    datasets. It combines data from publicly available sources and deep
                    learning-based hydrological modeling outputs.
                  </p>
                  <ul>
                    <li>
                      <strong>Public Datasets:</strong> CDL, NLCD, MODIS, LOCA2, PRISM, Wellogic
                    </li>
                    <li>
                      <strong>Hydrological Modeling Outputs:</strong> SWAT+, MODFLOW-generated
                      recharge estimates
                    </li>
                    <li>
                      <strong>Groundwater Data:</strong> Wellogic well records, EBK-interpolated
                      hydraulic properties
                    </li>
                    <li>
                      <strong>Land Cover & Terrain Features:</strong> Derived from MODIS, NLCD,
                      LANDFIRE, and DEM
                    </li>
                    <li>
                      <strong>Availability:</strong> Currently limited to Michigan Lower Peninsula
                    </li>
                  </ul>
                  <p>
                    Users can extract environmental and groundwater attributes based on spatial
                    queries, enabling research in hydrology, groundwater management, and climate
                    impact analysis.
                  </p>
                </InfoContent>
              </InfoSection>
            </CollapsibleContent>
          )}
        </Collapsible>

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
            <DataResults>
              <pre>{JSON.stringify(data, null, 2)}</pre>
            </DataResults>
          </DataDisplay>
        )}
      </Sidebar>

      <MapContainer>
        <MapComponent setFormData={setFormData} onGeometryChange={handleGeometryChange} />
      </MapContainer>
    </PageLayout>
  );
};

export default HydroGeoDataset;
