import React from 'react';

const LogoutForm = ({ handleConfirmLogout, handleCancelLogout }) => {
  return (
    <div className="logout-buttons text-center">
      <button
        id="confirm-logout"
        className="logout-btn btn btn-danger"
        onClick={handleConfirmLogout}
      >
        Yes, Logout
      </button>
      <button
        id="cancel-logout"
        className="cancel-btn btn btn-secondary"
        onClick={handleCancelLogout}
      >
        Cancel
      </button>
    </div>
  );
};

export default LogoutForm;
