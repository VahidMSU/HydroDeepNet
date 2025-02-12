import React from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';

// Styled Components
const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 50vh;
  padding: 2rem;
`;

// Logout Form Component
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

// Main Logout Component
const Logout = () => {
  const navigate = useNavigate();

  const handleConfirmLogout = async () => {
    try {
      const response = await fetch('/api/logout', {
        method: 'POST',
        credentials: 'include',
      });

      const data = await response.json();
      if (data.status === 'success') {
        console.log('Logout successful:', data.message);
        navigate('/login');
      } else {
        console.error('Logout failed:', data.message);
      }
    } catch (error) {
      console.error('Error during logout:', error);
      alert('Failed to logout. Please try again.');
    }
  };

  const handleCancelLogout = () => {
    navigate(-1);
  };

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

export default Logout;
