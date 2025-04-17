import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { GlobalStyle, Sidebar, NavLink, ViewDiv, Button } from '../../styles/Layout.tsx';

const LayoutTemplate = ({ children, handleLogout }) => {
  const navigate = useNavigate(); // Hook for navigation
  const [isSidebarVisible, setIsSidebarVisible] = useState(true); // State to toggle sidebar visibility

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (!token) {
      navigate('/login');
    }
  }, [navigate]);

  return (
    <>
      <GlobalStyle />
      {localStorage.getItem('authToken') && (
        <>
          <button
            onClick={() => setIsSidebarVisible(!isSidebarVisible)}
            style={{
              position: 'absolute',
              top: '10px',
              left: '10px',
              zIndex: 1000,
              padding: '5px 10px',
              background: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            {isSidebarVisible ? 'Hide Sidebar' : 'Show Sidebar'}
          </button>
          {isSidebarVisible && (
            <Sidebar>
              <h2>Navigation</h2>
              <nav>
                <NavLink to="/">
                  <i className="fas fa-home"></i> Home
                </NavLink>
                <NavLink to="/model_settings">
                  <i className="fas fa-cogs"></i> SWATGenX
                </NavLink>
                <NavLink to="/vision_system">
                  <i className="fas fa-eye"></i> Vision System
                </NavLink>
                <NavLink to="/hydro_geo_dataset">
                  <i className="fas fa-database"></i> HydroGeoDataset
                </NavLink>
                <NavLink to="/user_dashboard">
                  <i className="fas fa-tachometer-alt"></i> User Dashboard
                </NavLink>
                <Button onClick={handleLogout}>
                  <i className="fas fa-sign-out-alt"></i> Logout
                </Button>
                <NavLink to="/contact">
                  <i className="fas fa-envelope"></i> Contact
                </NavLink>
                <NavLink to="/about">
                  <i className="fas fa-info-circle"></i> About
                </NavLink>
                <NavLink to="/signup">
                  <i className="fas fa-user-plus"></i> Sign Up
                </NavLink>
              </nav>
            </Sidebar>
          )}
        </>
      )}
      <ViewDiv>{children}</ViewDiv>
      <footer>
        <Link to="/privacy">Privacy Policy</Link> | <Link to="/terms">Terms of Service</Link>
      </footer>
    </>
  );
};

export default LayoutTemplate;
