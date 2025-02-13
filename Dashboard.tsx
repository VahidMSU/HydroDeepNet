import React from 'react';
import { ContainerFluid, HeaderTitle } from './web_application/frontend/src/styles/SWATGenX.tsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTachometerAlt } from '@fortawesome/free-solid-svg-icons';

const Dashboard = () => {
  return (
    <ContainerFluid>
      <HeaderTitle>
        <FontAwesomeIcon icon={faTachometerAlt} />
        Dashboard
      </HeaderTitle>
      <div>
        {/* Dashboard content goes here */}
        <h2>Welcome to your dashboard!</h2>
        <p>This is where you can view your models and other information.</p>
      </div>
    </ContainerFluid>
  );
};

export default Dashboard;
