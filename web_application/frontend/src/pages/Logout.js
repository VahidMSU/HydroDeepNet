import React from 'react';
import { useNavigate } from 'react-router-dom';
import LogoutTemplate from '../components/templates/Logout'; // Import the new LogoutTemplate component
import '../styles/Logout.tsx'; // Adjust the path if necessary

const Logout = () => {
  const navigate = useNavigate();

  const handleConfirmLogout = async () => {
    try {
      const response = await fetch('/api/logout', {
        method: 'POST', // Ensure this matches the backend method
        credentials: 'include', // Ensures cookies are sent
      });

      const data = await response.json();
      if (data.status === 'success') {
        console.log('Logout successful:', data.message);
        navigate('/login'); // Redirect user to login
      } else {
        console.error('Logout failed:', data.message);
      }
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
    <LogoutTemplate
      handleConfirmLogout={handleConfirmLogout}
      handleCancelLogout={handleCancelLogout}
    />
  );
};

export default Logout;
