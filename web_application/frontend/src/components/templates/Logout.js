import React from 'react';
import LogoutForm from '../forms/LogoutForm'; // Import the new component

const LogoutTemplate = ({ handleConfirmLogout, handleCancelLogout }) => {
  return (
    <main className="container my-5">
      <h1 className="text-center">Logout</h1>
      <p className="text-center">Are you sure you want to log out?</p>
      <LogoutForm
        handleConfirmLogout={handleConfirmLogout}
        handleCancelLogout={handleCancelLogout}
      />
    </main>
  );
};

export default LogoutTemplate;
