import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import SWATGenXTemplate from '../components/templates/SWATGenX';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft } from '@fortawesome/free-solid-svg-icons';

// Prevent scrolling on this specific page and hide the sidebar
const SWATGenX = () => {
  const navigate = useNavigate();

  // Replace the sessionStorage refresh with a URL hash based refresh
  useEffect(() => {
    if (!window.location.hash.includes('reloaded')) {
      window.location.hash = 'reloaded';
      window.location.reload();
    }
  }, []);

  // Add effect to prevent body scrolling when this component mounts
  useEffect(() => {
    // Save original styles
    const originalStyle = window.getComputedStyle(document.body).overflow;

    // Prevent scrolling on body
    document.body.style.overflow = 'hidden';

    // Restore original style when component unmounts
    return () => {
      document.body.style.overflow = originalStyle;
    };
  }, []);

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
        left: '60px', // Adjusted position to avoid overlap
        zIndex: 1001,
        padding: '8px 15px',
        background: '#ff8500',
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
      <SWATGenXTemplate />
    </>
  );
};

export default SWATGenX;
