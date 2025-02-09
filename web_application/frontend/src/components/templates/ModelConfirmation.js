import React from 'react';
import { Link } from 'react-router-dom';
import '../../styles/ModelConfirmation.tsx'; // Create this file with the styles below

const ModelConfirmationTemplate = () => {
  return (
    <div className="container mt-5">
      <div className="card p-4">
        <div className="alert alert-success mb-0" role="alert">
          <h4 className="alert-heading">Model is Running!</h4>
          <p>
            Your model settings have been successfully submitted. The model is now being processed.
          </p>
          <hr />
          <p className="mb-0">
            You will be notified once the process is complete. Feel free to navigate through other
            sections in the meantime.
          </p>
        </div>
      </div>

      <div className="text-center mt-4">
        <Link to="/" className="btn btn-primary btn-lg">
          <i className="bi bi-arrow-left"></i> Back to Home
        </Link>
      </div>
    </div>
  );
};

export default ModelConfirmationTemplate;
