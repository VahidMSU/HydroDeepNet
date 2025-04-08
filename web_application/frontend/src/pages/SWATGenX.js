import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import SWATGenXTemplate from '../components/templates/SWATGenX';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft } from '@fortawesome/free-solid-svg-icons';
import '../styles/NoScroll.css'; // Import the no-scroll CSS

// Prevent scrolling on this specific page and hide the sidebar
const SWATGenX = () => {
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
    
    // Store that we're coming from a no-scroll page
    sessionStorage.setItem('came_from_noscroll', 'true');
    
    // Restore original state when component unmounts
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
        left: '60px', // Adjusted position to avoid overlap
        zIndex: 1001,
        padding: '8px 15px',
        background: '#222222', // changed to dark gray (#222222)
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
    <div className="no-scroll-container">
      <ReturnButton />
      <SWATGenXTemplate />
    </div>
  );
};

export default SWATGenX;
