import React, { useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  ContainerFluid,
  Sidebar,
  Nav,
  NavLink,
  ContentWrapper,
  Footer,
  LogoutButton,
} from '../../styles/LayoutTemplate.tsx';

const LayoutTemplate = ({ children, handleLogout }) => {
  const navigate = useNavigate(); // Hook for navigation

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (!token) {
      navigate('/login');
    }
  }, [navigate]);

  return (
    <ContainerFluid>
      {localStorage.getItem('authToken') && (
        <Sidebar>
          <h2>Navigation</h2>
          <Nav>
            <NavLink to="/">
              <i className="fas fa-home"></i> Home
            </NavLink>
            <NavLink to="/model_settings">
              <i className="fas fa-cogs"></i> Model Settings
            </NavLink>
            <NavLink to="/visualizations">
              <i className="fas fa-chart-bar"></i> Visualizations
            </NavLink>
            <NavLink to="/michigan">
              <i className="fas fa-map-marker-alt"></i> Michigan
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
            <LogoutButton onClick={handleLogout}>
              <i className="fas fa-sign-out-alt"></i> Logout
            </LogoutButton>
            <NavLink to="/contact">
              <i className="fas fa-envelope"></i> Contact
            </NavLink>
            <NavLink to="/about">
              <i className="fas fa-info-circle"></i> About
            </NavLink>
            <NavLink to="/signup">
              <i className="fas fa-user-plus"></i> Sign Up
            </NavLink>
          </Nav>
        </Sidebar>
      )}
      <ContentWrapper>{children}</ContentWrapper>
      <Footer>
        <Link to="/privacy">Privacy Policy</Link> | <Link to="/terms">Terms of Service</Link>
      </Footer>
    </ContainerFluid>
  );
};

export default LayoutTemplate;
