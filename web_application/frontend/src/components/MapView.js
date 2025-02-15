// components/MapView.js

import React, { useEffect, useRef } from 'react';
import MapView from '@arcgis/core/views/MapView';
import Map from '@arcgis/core/Map';
import Graphic from '@arcgis/core/Graphic';
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer';

function EsriMap({ stationData }) {
  const mapRef = useRef(null);

  useEffect(() => {
    const map = new Map({ basemap: 'topo-vector' });
    const view = new MapView({
      container: mapRef.current,
      map: map,
      zoom: 4,
      center: [-90, 38],
    });

    const graphicsLayer = new GraphicsLayer();
    map.add(graphicsLayer);

    if (stationData?.Latitude && stationData?.Longitude) {
      view.goTo({
        center: [stationData.Longitude, stationData.Latitude],
        zoom: 10,
      });

      const point = {
        type: 'point',
        longitude: stationData.Longitude,
        latitude: stationData.Latitude,
      };

      const marker = new Graphic({
        geometry: point,
        symbol: { type: 'simple-marker', color: [226, 119, 40] },
      });

      graphicsLayer.add(marker);
    }

    return () => {
      view.destroy();
    };
  }, [stationData]);

  return <div ref={mapRef} style={{ height: '500px', border: '1px solid #ccc' }} />;
}

export default EsriMap;
