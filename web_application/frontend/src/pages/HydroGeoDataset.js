import React from 'react';
import { useNavigate } from 'react-router-dom';
import HydroGeoDatasetTemplate from '../components/templates/HydroGeoDataset.js';
import '../styles/Layout.tsx'; // Ensure the path is correct
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft } from '@fortawesome/free-solid-svg-icons';

const HydroGeoDataset = () => {
  const navigate = useNavigate();

  const handleReturn = () => {
    navigate('/');
  };

  // Return button to navigate back to home page
  const ReturnButton = () => (
    <button
      onClick={handleReturn}
      style={{
        position: 'absolute',
        top: '10px',
        left: '60px', // Positioned to avoid overlap with sidebar toggle
        zIndex: 1001,
        padding: '8px 15px',
        background: '#222222',
        color: 'white',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)',
      }}
    >
      <FontAwesomeIcon icon={faArrowLeft} />
      Return to Home
    </button>
  );

  return (
    <>
      <ReturnButton />
      <HydroGeoDatasetTemplate />
    </>
  );
};

export default HydroGeoDataset;
