import React from 'react';
import { Link } from 'react-router-dom';
import { Body, Card, BtnPrimary } from '../../styles/ModelConfirmation.tsx'; // Import styled components

const ModelConfirmationTemplate = () => {
  return (
    <Body>
      <Card>
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
      </Card>

      <div className="text-center mt-4">
        <Link to="/">
          <BtnPrimary>
            <i className="bi bi-arrow-left"></i> Back to Home
          </BtnPrimary>
        </Link>
      </div>
    </Body>
  );
};

export default ModelConfirmationTemplate;
