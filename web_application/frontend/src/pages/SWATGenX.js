import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import SWATGenXTemplate from '../components/templates/SWATGenX';
import '../styles/NoScroll.css'; // Import the no-scroll CSS

// Prevent scrolling on this specific page
const SWATGenX = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  
  // Function to ensure sidebar is visible
  const ensureSidebarVisible = useCallback(() => {
    const sidebarElement = document.querySelector('.MuiDrawer-paper');
    
    // If sidebar is hidden, dispatch event to show it
    if (!sidebarElement || window.getComputedStyle(sidebarElement).display === 'none') {
      // Set state directly rather than using events
      setIsSidebarVisible(true);
      
      // Also dispatch a direct event for other listeners
      window.dispatchEvent(
        new CustomEvent('sidebar-toggle-direct', {
          detail: { visible: true }
        })
      );
    }
    
    // Highlight the button, but don't modify its functionality
    const closeButton = document.getElementById('sidebar-toggle-button');
    
    if (closeButton) {
      // Only add highlighting
      closeButton.classList.add('highlight-button');
      
      // Remove the highlighting after a short delay
      setTimeout(() => {
        closeButton.classList.remove('highlight-button');
      }, 1500);
    }
  }, []);

  // Add effect to manage scroll locking and listen for sidebar events
  useEffect(() => {
    // Add the no-scroll classes
    document.documentElement.classList.add('no-scroll-page');
    document.body.classList.add('no-scroll-page');
    
    // Store that we're coming from a no-scroll page
    sessionStorage.setItem('came_from_noscroll', 'true');
    
    // Simple direct sidebar check - more reliable
    const checkSidebar = () => {
      const sidebarElement = document.querySelector('.MuiDrawer-paper');
      const isVisible = !!sidebarElement && window.getComputedStyle(sidebarElement).display !== 'none';
      setIsSidebarVisible(isVisible);
    };
    
    // Initial check
    checkSidebar();
    
    // Direct event listeners for sidebar toggle
    const handleDirectSidebarToggle = (event) => {
      if (event.detail && event.detail.visible !== undefined) {
        setIsSidebarVisible(event.detail.visible);
      }
    };
    
    // Listen for direct toggle events
    window.addEventListener('sidebar-toggle-direct', handleDirectSidebarToggle);
    
    // Also add a direct click handler to the button as a fallback
    const toggleButton = document.getElementById('sidebar-toggle-button');
    if (toggleButton) {
      // Force the button to use our handler
      const originalClickHandler = toggleButton.onclick;
      toggleButton.onclick = (e) => {
        originalClickHandler(e);
        // Double-check sidebar state after a short delay
        setTimeout(checkSidebar, 50);
      };
    }
    
    // Add CSS for the button animation
    const style = document.createElement('style');
    style.textContent = `
      @keyframes pulse-button {
        0% { box-shadow: 0 0 0 0 rgba(255, 133, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 133, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 133, 0, 0); }
      }
      .highlight-button {
        animation: pulse-button 1s infinite;
        background-color: rgba(255, 133, 0, 0.7) !important;
      }
    `;
    document.head.appendChild(style);
    
    // Ensure sidebar is visible on initial load - simpler approach
    setTimeout(ensureSidebarVisible, 100);
    
    // Restore original state when component unmounts
    return () => {
      // Clean up all event listeners and timeouts
      window.removeEventListener('sidebar-toggle-direct', handleDirectSidebarToggle);
      
      document.documentElement.classList.remove('no-scroll-page');
      document.body.classList.remove('no-scroll-page');
      
      if (style && document.head.contains(style)) {
        document.head.removeChild(style);
      }
    };
  }, [ensureSidebarVisible]);

  const sidebarWidth = 250; // This should match the drawerWidth from Layout.js

  return (
    <div 
      className="no-scroll-container swatgenx-page-container"
      style={{
        marginLeft: isSidebarVisible ? `${sidebarWidth}px` : '0',
        width: isSidebarVisible ? `calc(100vw - ${sidebarWidth}px)` : '100vw',
        maxWidth: isSidebarVisible ? `calc(100vw - ${sidebarWidth}px)` : '100vw',
        transition: 'margin-left 0.3s ease, width 0.3s ease',
        position: 'fixed',
        top: 0,
        right: 0,
        left: isSidebarVisible ? `${sidebarWidth}px` : '0',
        height: '100vh',
        maxHeight: '100vh',
        overflow: 'hidden',
        boxSizing: 'border-box'
      }}
      onClick={ensureSidebarVisible}
    >
      <div style={{
        width: '100%',
        height: '100%',
        maxHeight: '100vh',
        position: 'relative',
        overflow: 'hidden',
        paddingTop: '10px',
        paddingRight: '10px',
        display: 'flex',
        flexDirection: 'column',
        boxSizing: 'border-box'
      }}>
        <SWATGenXTemplate />
      </div>
    </div>
  );
};

export default SWATGenX;
