import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import HydroGeoDatasetTemplate from '../components/templates/HydroGeoDataset.js';
import '../styles/Layout.tsx'; // Ensure the path is correct
import '../styles/NoScroll.css'; // Import the no-scroll CSS
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft } from '@fortawesome/free-solid-svg-icons';

const HydroGeoDataset = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Force a clean state by adding a URL hash for loading
  useEffect(() => {
    if (!window.location.hash.includes('noscroll_loaded')) {
      window.location.hash = 'noscroll_loaded';
      window.location.reload();
    }
  }, []);

  // Add effect to manage scroll locking
  useEffect(() => {
    // Add the no-scroll classes
    document.documentElement.classList.add('no-scroll-page');
    document.body.classList.add('no-scroll-page');

    // Store the previous path to restore it later
    sessionStorage.setItem('came_from_noscroll', 'true');
    
    // Cleanup function to restore original styles
    return () => {
      document.documentElement.classList.remove('no-scroll-page');
      document.body.classList.remove('no-scroll-page');
    };
  }, []);

  const handleReturn = () => {
    // Remove the noscroll marker before navigating
    sessionStorage.removeItem('came_from_noscroll');
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
    <div className="no-scroll-container no-scroll-with-sidebar">
      <ReturnButton />
      <HydroGeoDatasetTemplate />
    </div>
  );
};

export default HydroGeoDataset;
