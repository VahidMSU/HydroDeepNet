import React from 'react';

const HydroGeoDatasetForm = ({
  formData,
  handleChange,
  handleSubmit,
  availableVariables,
  availableSubvariables,
}) => {
  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="variable">Variable:</label>
        <select
          id="variable"
          name="variable"
          className="form-control"
          value={formData.variable}
          onChange={handleChange}
        >
          <option value="">Select Variable</option>
          {availableVariables.map((variable) => (
            <option key={variable} value={variable}>
              {variable}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="subvariable">Subvariable:</label>
        <select
          id="subvariable"
          name="subvariable"
          className="form-control"
          value={formData.subvariable}
          onChange={handleChange}
        >
          <option value="">Select Subvariable</option>
          {availableSubvariables.map((subvariable) => (
            <option key={subvariable} value={subvariable}>
              {subvariable}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="latitude">Latitude:</label>
        <input
          type="text"
          id="latitude"
          name="latitude"
          className="form-control"
          value={formData.latitude}
          onChange={handleChange}
        />
      </div>

      <div className="form-group">
        <label htmlFor="longitude">Longitude:</label>
        <input
          type="text"
          id="longitude"
          name="longitude"
          className="form-control"
          value={formData.longitude}
          onChange={handleChange}
        />
      </div>

      <div className="form-group">
        <label htmlFor="min_latitude">Min Latitude:</label>
        <input
          type="text"
          id="min_latitude"
          name="min_latitude"
          className="form-control"
          value={formData.min_latitude}
          onChange={handleChange}
        />
      </div>

      <div className="form-group">
        <label htmlFor="max_latitude">Max Latitude:</label>
        <input
          type="text"
          id="max_latitude"
          name="max_latitude"
          className="form-control"
          value={formData.max_latitude}
          onChange={handleChange}
        />
      </div>

      <div className="form-group">
        <label htmlFor="min_longitude">Min Longitude:</label>
        <input
          type="text"
          id="min_longitude"
          name="min_longitude"
          className="form-control"
          value={formData.min_longitude}
          onChange={handleChange}
        />
      </div>

      <div className="form-group">
        <label htmlFor="max_longitude">Max Longitude:</label>
        <input
          type="text"
          id="max_longitude"
          name="max_longitude"
          className="form-control"
          value={formData.max_longitude}
          onChange={handleChange}
        />
      </div>

      <button type="submit" className="btn btn-primary">
        Fetch Data
      </button>
    </form>
  );
};

export default HydroGeoDatasetForm;
