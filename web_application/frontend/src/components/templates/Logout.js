import React from 'react';
import LogoutForm from '../forms/LogoutForm'; // Import the new component
import { Container } from '../../styles/Logout.tsx'; // Import styled components

const LogoutTemplate = ({ handleConfirmLogout, handleCancelLogout }) => {
  return (
    <Container>
      <h1>Logout</h1>
      <p>Are you sure you want to log out?</p>
      <LogoutForm
        handleConfirmLogout={handleConfirmLogout}
        handleCancelLogout={handleCancelLogout}
      />
    </Container>
  );
};

export default LogoutTemplate;
