import React, { useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Box,
  Typography,
  Divider,
} from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import SettingsIcon from '@mui/icons-material/Settings';
import BarChartIcon from '@mui/icons-material/BarChart';
import PlaceIcon from '@mui/icons-material/Place';
import VisibilityIcon from '@mui/icons-material/Visibility';
import StorageIcon from '@mui/icons-material/Storage';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import EmailIcon from '@mui/icons-material/Email';
import InfoIcon from '@mui/icons-material/Info';

const drawerWidth = 250;

const Layout = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (!token) {
      navigate('/login');
    }
  }, [navigate]);

  const handleLogout = async () => {
    try {
      const response = await fetch('/api/logout', {
        method: 'POST',
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Logout failed');
      }

      localStorage.removeItem('authToken');
      navigate('/login');
    } catch (error) {
      console.error('Error during logout:', error);
      alert('Failed to logout. Please try again.');
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        bgcolor: '#2b2b2c',
        minHeight: '100vh',
        width: '100vw',
        color: 'white',
      }}
    >
      {localStorage.getItem('authToken') && (
        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: drawerWidth,
              boxSizing: 'border-box',
              bgcolor: '#2b2b2c',
              color: 'white',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
            },
          }}
        >
          <Box>
            <Toolbar />
            <Typography
              variant="h5"
              sx={{ color: 'white', textAlign: 'center', mt: 2, mb: 2, fontWeight: 'bold' }}
            >
              HydroDeepNet
            </Typography>
            <List>
              {[
                { text: 'Home', icon: <HomeIcon />, path: '/' },
                { text: 'SWATGenX', icon: <SettingsIcon />, path: '/model_settings' },
                { text: 'Visualizations', icon: <BarChartIcon />, path: '/visualizations' },
                { text: 'Michigan', icon: <PlaceIcon />, path: '/michigan' },
                { text: 'Vision System', icon: <VisibilityIcon />, path: '/vision_system' },
                { text: 'HydroGeoDataset', icon: <StorageIcon />, path: '/hydro_geo_dataset' },
                { text: 'User Dashboard', icon: <DashboardIcon />, path: '/user_dashboard' },
                { text: 'Contact', icon: <EmailIcon />, path: '/contact' },
                { text: 'About', icon: <InfoIcon />, path: '/about' },
                //{ text: 'Sign Up', icon: <PersonAddIcon />, path: '/signup' },
              ].map((item) => (
                <ListItemButton
                  key={item.text}
                  component={Link}
                  to={item.path}
                  selected={location.pathname === item.path}
                  sx={{
                    color: 'white',
                    '&.Mui-selected': {
                      backgroundColor: '#687891',
                      color: 'white',
                      fontWeight: 'bold',
                    },
                    '&:hover': {
                      color: '#ff8500',
                    },
                  }}
                >
                  <ListItemIcon sx={{ color: 'inherit' }}>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              ))}
            </List>
          </Box>

          {/* Logout Section */}
          <Box sx={{ width: '100%' }}>
            <Divider sx={{ bgcolor: '#687891', mx: 2 }} />
            <List>
              <ListItemButton
                onClick={handleLogout}
                sx={{ color: 'white', '&:hover': { color: '#ff8500' } }}
              >
                <ListItemIcon>
                  <ExitToAppIcon sx={{ color: 'white' }} />
                </ListItemIcon>
                <ListItemText primary="Logout" />
              </ListItemButton>
            </List>
          </Box>

          {/* Footer Links */}
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" sx={{ color: '#687891' }}>
              <Link to="/privacy" style={{ textDecoration: 'none', color: 'inherit' }}>
                Privacy Policy
              </Link>{' '}
              |
              <Link to="/terms" style={{ textDecoration: 'none', color: 'inherit', marginLeft: 8 }}>
                Terms of Service
              </Link>
            </Typography>
          </Box>
        </Drawer>
      )}

      {/* Main Content */}
      <Box component="main" sx={{ flexGrow: 1, p: 3, minHeight: '100vh' }}>
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default Layout;

// Backup of Original

// import React, { useEffect } from 'react';
// import { Link, useNavigate } from 'react-router-dom';
// import '../styles/Layout.tsx'; // Ensure correct path for CSS

// const Layout = ({ children }) => {
//   const navigate = useNavigate(); // Hook for navigation

//   useEffect(() => {
//     const token = localStorage.getItem('authToken');
//     if (!token) {
//       navigate('/login');
//     }
//   }, [navigate]);

//   const handleLogout = async () => {
//     try {
//       const response = await fetch('/api/logout', {
//         method: 'POST', // Ensure method is 'POST'
//         credentials: 'include', // Include cookies for session handling
//       });

//       if (!response.ok) {
//         throw new Error('Logout failed');
//       }

//       console.log('User logged out successfully');
//       // Clear token and redirect to the login page after logout
//       localStorage.removeItem('authToken');
//       navigate('/login');
//     } catch (error) {
//       console.error('Error during logout:', error);
//       alert('Failed to logout. Please try again.');
//     }
//   };

//   return (
//     <div className="container-fluid">
//       {localStorage.getItem('authToken') && (
//         <div className="sidebar">
//           <h2>Navigation</h2>
//           <nav className="nav flex-column">
//             <Link className="nav-link" to="/">
//               <i className="fas fa-home"></i> Home
//             </Link>
//             <Link className="nav-link" to="/model_settings">
//               <i className="fas fa-cogs"></i> SWATGenX
//             </Link>
//             <Link className="nav-link" to="/visualizations">
//               <i className="fas fa-chart-bar"></i> Visualizations
//             </Link>
//             <Link className="nav-link" to="/michigan">
//               <i className="fas fa-map-marker-alt"></i> Michigan
//             </Link>
//             <Link className="nav-link" to="/vision_system">
//               <i className="fas fa-eye"></i> Vision System
//             </Link>
//             <Link className="nav-link" to="/hydro_geo_dataset">
//               <i className="fas fa-database"></i> HydroGeoDataset
//             </Link>
//             <Link className="nav-link" to="/user_dashboard">
//               <i className="fas fa-tachometer-alt"></i> User Dashboard
//             </Link>
//             {/* Logout Button */}
//             <button className="nav-link btn btn-link text-start" onClick={handleLogout}>
//               <i className="fas fa-sign-out-alt"></i> Logout
//             </button>
//             <Link className="nav-link" to="/contact">
//               <i className="fas fa-envelope"></i> Contact
//             </Link>
//             <Link className="nav-link" to="/about">
//               <i className="fas fa-info-circle"></i> About
//             </Link>
//             <Link className="nav-link" to="/signup">
//               <i className="fas fa-user-plus"></i> Sign Up
//             </Link>
//           </nav>
//         </div>
//       )}
//       {/* Main Content Area */}
//       <div className="content-wrapper">{children}</div>
//       <footer className="footer mt-4 p-3 text-center border-top">
//         <Link to="/privacy">Privacy Policy</Link> | <Link to="/terms">Terms of Service</Link>
//       </footer>
//     </div>
//   );
// };

// export default Layout;
