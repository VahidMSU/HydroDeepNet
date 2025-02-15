import React from 'react';

const FormComponent = ({ formData, setFormData, data, setData }) => {
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const dummyData = {
      variable: formData.variable,
      subvariable: formData.subvariable,
      result: 'Sample fetched value',
    };
    setData(dummyData);
  };

  return (
    <div className="form-container">
      <div className="form-card">
        <form onSubmit={handleSubmit}>
          {/* Single Point Inputs */}
          <div className="mb-3">
            <label htmlFor="latitude" className="form-label">
              Latitude
            </label>
            <input
              type="text"
              className="form-control"
              id="latitude"
              name="latitude"
              value={formData.latitude}
              readOnly
            />
          </div>
          <div className="mb-3">
            <label htmlFor="longitude" className="form-label">
              Longitude
            </label>
            <input
              type="text"
              className="form-control"
              id="longitude"
              name="longitude"
              value={formData.longitude}
              readOnly
            />
          </div>
          <hr />

          {/* Range Inputs */}
          <div className="mb-3">
            <label htmlFor="min_latitude" className="form-label">
              Min Latitude
            </label>
            <input
              type="text"
              className="form-control"
              id="min_latitude"
              name="min_latitude"
              value={formData.min_latitude}
              readOnly
            />
          </div>
          <div className="mb-3">
            <label htmlFor="max_latitude" className="form-label">
              Max Latitude
            </label>
            <input
              type="text"
              className="form-control"
              id="max_latitude"
              name="max_latitude"
              value={formData.max_latitude}
              readOnly
            />
          </div>
          <div className="mb-3">
            <label htmlFor="min_longitude" className="form-label">
              Min Longitude
            </label>
            <input
              type="text"
              className="form-control"
              id="min_longitude"
              name="min_longitude"
              value={formData.min_longitude}
              readOnly
            />
          </div>
          <div className="mb-3">
            <label htmlFor="max_longitude" className="form-label">
              Max Longitude
            </label>
            <input
              type="text"
              className="form-control"
              id="max_longitude"
              name="max_longitude"
              value={formData.max_longitude}
              readOnly
            />
          </div>

          {/* Variable and Subvariable Inputs */}
          <div className="mb-3">
            <label htmlFor="variable" className="form-label">
              Variable
            </label>
            <select
              className="form-select"
              id="variable"
              name="variable"
              value={formData.variable}
              onChange={handleChange}
            >
              <option value="">Select Variable</option>
              <option value="var1">Variable 1</option>
              <option value="var2">Variable 2</option>
              {/* Add more options as needed */}
            </select>
          </div>
          <div className="mb-3">
            <label htmlFor="subvariable" className="form-label">
              Subvariable
            </label>
            <select
              className="form-select"
              id="subvariable"
              name="subvariable"
              value={formData.subvariable}
              onChange={handleChange}
            >
              <option value="">Select Subvariable</option>
              <option value="sub1">Subvariable 1</option>
              <option value="sub2">Subvariable 2</option>
              {/* Add more options as needed */}
            </select>
          </div>

          {/* Submit Button */}
          <button type="submit" className="btn btn-primary w-100">
            Get Variable Value
          </button>
        </form>

        {/* Display Results */}
        {data && (
          <div id="result" className="mt-4">
            <h3 className="text-center">Fetched Data</h3>
            <pre className="border rounded p-3 bg-light">{JSON.stringify(data, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default FormComponent;
