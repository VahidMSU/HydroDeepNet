import React from 'react';

const LogoutForm = ({ handleConfirmLogout, handleCancelLogout }) => {
  return (
    <div className="text-center">
      <button className="btn btn-danger me-2" onClick={handleConfirmLogout}>
        Confirm Logout
      </button>
      <button className="btn btn-secondary" onClick={handleCancelLogout}>
        Cancel
      </button>
    </div>
  );
};

export default LogoutForm;
