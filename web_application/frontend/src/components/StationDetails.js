// componenets/StationDetails.js
import React from 'react';

function StationDetails({ stationData }) {
  if (!stationData) {
    return null;
  }

  return (
    <div className="card mt-3">
      <div className="card-body">
        <h5>Station Characteristics</h5>
        <ul>
          {Object.entries(stationData) // ✅ Convert Object to Array
            .filter(([key]) => key !== 'geometries') // ✅ Ignore geometries
            .filter(([key]) => key !== 'streams_geometries') // ✅ Ignore streams_geometries
            .filter(([key]) => key !== 'lakes_geometries') // ✅ Ignore lakes_geometries

            .map(([key, value]) => (
              <li key={key}>
                <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value) : value}
              </li>
            ))}
        </ul>
      </div>
    </div>
  );
}

export default StationDetails;
