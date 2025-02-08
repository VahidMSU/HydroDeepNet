import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import '../css/Layout.css'; // Ensure correct path for CSS

const Layout = ({ children }) => {
  const navigate = useNavigate(); // Hook for navigation

  const handleLogout = async () => {
    try {
      const response = await fetch('/api/logout', {
        method: 'POST', // Ensure method is 'POST'
        credentials: 'include', // Include cookies for session handling
      });

      if (!response.ok) {
        throw new Error('Logout failed');
      }

      console.log('User logged out successfully');
      // Redirect to the login page after logout
      navigate('/login');
    } catch (error) {
      console.error('Error during logout:', error);
      alert('Failed to logout. Please try again.');
    }
  };

  return (
    <div className="container-fluid">
      <div className="sidebar">
        <h2>Navigation</h2>
        <nav className="nav flex-column">
          <Link className="nav-link" to="/">
            <i className="fas fa-home"></i> Home
          </Link>
          <Link className="nav-link" to="/model_settings">
            <i className="fas fa-cogs"></i> Model Settings
          </Link>
          <Link className="nav-link" to="/visualizations">
            <i className="fas fa-chart-bar"></i> Visualizations
          </Link>
          <Link className="nav-link" to="/michigan">
            <i className="fas fa-map-marker-alt"></i> Michigan
          </Link>
          <Link className="nav-link" to="/vision_system">
            <i className="fas fa-eye"></i> Vision System
          </Link>
          <Link className="nav-link" to="/hydro_geo_dataset">
            <i className="fas fa-database"></i> HydroGeoDataset
          </Link>
          <Link className="nav-link" to="/user_dashboard">
            <i className="fas fa-tachometer-alt"></i> User Dashboard
          </Link>
          {/* Logout Button */}
          <button className="nav-link btn btn-link text-start" onClick={handleLogout}>
            <i className="fas fa-sign-out-alt"></i> Logout
          </button>
          <Link className="nav-link" to="/contact">
            <i className="fas fa-envelope"></i> Contact
          </Link>
          <Link className="nav-link" to="/about">
            <i className="fas fa-info-circle"></i> About
          </Link>
          <Link className="nav-link" to="/signup">
            <i className="fas fa-user-plus"></i> Sign Up
          </Link>
        </nav>
      </div>
      {/* Main Content Area */}
      <div className="content-wrapper">{children}</div>
      <footer className="footer mt-4 p-3 text-center border-top">
        <Link to="/privacy">Privacy Policy</Link> | <Link to="/terms">Terms of Service</Link>
      </footer>
    </div>
  );
};

export default Layout;
