import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../css/Logout.css'; // Adjust the path if necessary

const Logout = () => {
  const navigate = useNavigate();

  const handleConfirmLogout = async () => {
    try {
      // Make API call to logout endpoint
      const response = await fetch('logout', {
        method: 'GET',
        credentials: 'include', // Ensures cookies are sent
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.status === 'success') {
            console.log('Logout successful:', data.message);
            navigate('/login'); // Redirect user to login
          } else {
            console.error('Logout failed:', data.message);
          }
        })
        .catch((error) => {
          console.error('Error during logout:', error);
        });
    } catch (error) {
      console.error('Error during logout:', error);
      alert('Failed to logout. Please try again.');
    }
  };

  const handleCancelLogout = () => {
    // Navigate back to the previous page
    navigate(-1); // Go back one step in history
  };

  return (
    <main className="container my-5">
      <h1 className="text-center">Logout</h1>
      <p className="text-center">Are you sure you want to log out?</p>
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
    </main>
  );
};

export default Logout;
