import React from 'react';
import { Link } from 'react-router-dom';
import {
  ContainerFluid,
  Card,
  CardBody,
  HeaderTitle,
  ButtonPrimary,
} from '../../styles/Layout.tsx'; // Import styled components

const ModelConfirmationTemplate = () => {
  return (
    <ContainerFluid>
      <Card>
        <CardBody>
          <div className="alert alert-success mb-0" role="alert">
            <HeaderTitle>Model is Running!</HeaderTitle>
            <p>
              Your model settings have been successfully submitted. The model is now being
              processed.
            </p>
            <hr />
            <p className="mb-0">
              You will be notified once the process is complete. Feel free to navigate through other
              sections in the meantime.
            </p>
          </div>
        </CardBody>
      </Card>

      <div className="text-center mt-4">
        <Link to="/">
          <ButtonPrimary>
            <i className="bi bi-arrow-left"></i> Back to Home
          </ButtonPrimary>
        </Link>
      </div>
    </ContainerFluid>
  );
};

export default ModelConfirmationTemplate;
