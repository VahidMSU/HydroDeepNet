import React, { useEffect, useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import SessionChecker from '../components/SessionChecker';
import UserInfo from '../components/UserInfo';
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
  IconButton,
} from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import SettingsIcon from '@mui/icons-material/Settings';
import BarChartIcon from '@mui/icons-material/BarChart';
import PlaceIcon from '@mui/icons-material/Place';
import VisibilityIcon from '@mui/icons-material/Visibility';
import StorageIcon from '@mui/icons-material/Storage';
import FolderIcon from '@mui/icons-material/Folder';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import EmailIcon from '@mui/icons-material/Email';
import InfoIcon from '@mui/icons-material/Info';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import SmartToyIcon from '@mui/icons-material/SmartToy';

const drawerWidth = 250;

const Layout = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  const [userName, setUserName] = useState('');
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  useEffect(() => {
    // Check if we should hide the sidebar based on current route
    if (location.pathname === '/hydrogeo-assistant' || 
        location.pathname === '/hydro_geo_dataset' || 
        location.pathname === '/model_settings') {
      // For HydroGeoAssistant, HydroGeoDataset, and SWATGenX, we want to keep the sidebar visible
      setIsSidebarVisible(true);
    } else {
      // Check if sidebar was previously hidden
      const shouldHideSidebar = sessionStorage.getItem('hideSidebar') === 'true';
      if (!shouldHideSidebar) {
        setIsSidebarVisible(true);
      }
    }
  }, [location.pathname]);

  // Listen for custom events from HydroGeoAssistant and HydroGeoDataset
  useEffect(() => {
    const handleHydroGeoAssistantLoaded = (event) => {
      if (event.detail && event.detail.hideSidebar) {
        setIsSidebarVisible(false);
      }
    };

    const handleSidebarToggle = (event) => {
      if (event.detail !== undefined) {
        setIsSidebarVisible(event.detail.showSidebar);
      }
    };
    
    // Listen for our own sidebar toggle event
    const handleGlobalSidebarToggle = (event) => {
      if (event.detail && event.detail.visible !== undefined) {
        setIsSidebarVisible(event.detail.visible);
      }
    };

    // Add event listeners for the custom events
    window.addEventListener('hydrogeo-assistant-loaded', handleHydroGeoAssistantLoaded);
    window.addEventListener('hydrogeo-sidebar-toggle', handleSidebarToggle);
    window.addEventListener('sidebar-toggle', handleGlobalSidebarToggle);

    // Check sessionStorage on component mount
    if (sessionStorage.getItem('hideSidebar') === 'true') {
      setIsSidebarVisible(false);
    }

    // Cleanup
    return () => {
      window.removeEventListener('hydrogeo-assistant-loaded', handleHydroGeoAssistantLoaded);
      window.removeEventListener('hydrogeo-sidebar-toggle', handleSidebarToggle);
      window.removeEventListener('sidebar-toggle', handleGlobalSidebarToggle);
    };
  }, []);

  useEffect(() => {
    // Parse query params to check for OAuth redirect
    const queryParams = new URLSearchParams(location.search);
    const googleLogin = queryParams.get('google_login');
    const username = queryParams.get('username');
    
    // If this is a Google OAuth redirect, handle it
    if (googleLogin === 'success' && username) {
      console.log('Google OAuth redirect detected, setting auth state...');
      localStorage.setItem('authToken', 'true');
      localStorage.setItem('username', username);
      
      // Set username directly from redirect params
      setUserName(username);
      localStorage.setItem('userName', username);
      
      // Remove query parameters for cleaner URL
      navigate(location.pathname, { replace: true });
      setIsCheckingAuth(false);
      return;
    }

    // For non-OAuth login, proceed with normal auth check
    const checkAuth = async () => {
      setIsCheckingAuth(true);
      const token = localStorage.getItem('authToken');
      
      if (!token) {
        console.log('No auth token found, redirecting to login');
        // Small delay to prevent race conditions during page initialization
        setTimeout(() => {
          navigate('/login');
        }, 100);
      } else {
        const storedUserName = localStorage.getItem('userName') || localStorage.getItem('username');
        if (storedUserName && storedUserName !== 'User' && storedUserName !== 'undefined') {
          setUserName(storedUserName);
          console.log('Username found in localStorage:', storedUserName);
        } else {
          console.log('No username in localStorage, fetching from API');
          fetchUserInfo();
        }
      }
      setIsCheckingAuth(false);
    };
    
    checkAuth();
  }, [navigate, location]);

  const fetchUserInfo = async () => {
    try {
      console.log('Fetching user info from API...');
      const response = await fetch('/api/user/profile', {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('authToken')}`,
        },
        credentials: 'include',
      });

      if (response.ok) {
        const userData = await response.json();
        console.log('User data received:', userData);

        const name =
          userData.name ||
          userData.username ||
          userData.email ||
          userData.user?.name ||
          userData.user?.username ||
          userData.user?.email;

        if (name && name !== 'undefined') {
          console.log('Setting username to:', name);
          setUserName(name);
          localStorage.setItem('userName', name);
        } else {
          console.warn('No valid username found in user data');
          setUserName('User');
        }
      } else {
        console.error('Failed to fetch user profile:', response.status);
        fetchAlternateUserInfo();
      }
    } catch (error) {
      console.error('Error fetching user info:', error);
      fetchAlternateUserInfo();
    }
  };

  const fetchAlternateUserInfo = async () => {
    try {
      console.log('Trying alternate user info endpoint...');
      const response = await fetch('/api/auth/me', {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('authToken')}`,
        },
        credentials: 'include',
      });

      if (response.ok) {
        const userData = await response.json();
        console.log('Alternate user data received:', userData);

        const name =
          userData.name ||
          userData.username ||
          userData.email ||
          userData.user?.name ||
          userData.user?.username ||
          userData.user?.email;

        if (name && name !== 'undefined') {
          console.log('Setting username from alternate source to:', name);
          setUserName(name);
          localStorage.setItem('userName', name);
        }
      }
    } catch (error) {
      console.error('Error fetching alternate user info:', error);
    }
  };

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

  // Add new effect to adjust DOM for specific pages
  useEffect(() => {
    // Special handling for pages that need specific layout adjustments
    const isSpecialPage = location.pathname === '/hydro_geo_dataset' || 
                         location.pathname === '/hydrogeo-assistant' ||
                         location.pathname === '/model_settings';
    
    if (isSpecialPage) {
      // Add specific class to the body for targeting with CSS
      document.body.classList.add('hydrogeo-special-page');
      
      if (location.pathname === '/hydro_geo_dataset') {
        document.body.classList.add('hydrogeo-dataset-page');
      } else if (location.pathname === '/hydrogeo-assistant') {
        document.body.classList.add('hydrogeo-assistant-page');
      } else if (location.pathname === '/model_settings') {
        document.body.classList.add('swatgenx-page');
      }
      
      // Find the main content container and adjust it
      const mainContent = document.querySelector('.MuiBox-root');
      if (mainContent) {
        mainContent.style.paddingLeft = '0';
        mainContent.style.paddingRight = '0';
        mainContent.style.paddingTop = '0';
        mainContent.style.paddingBottom = '0';
      }
      
      // Force sidebar to be visible
      setIsSidebarVisible(true);
    } else {
      // Remove classes when navigating away
      document.body.classList.remove('hydrogeo-special-page');
      document.body.classList.remove('hydrogeo-dataset-page');
      document.body.classList.remove('hydrogeo-assistant-page');
      document.body.classList.remove('swatgenx-page');
    }
    
    return () => {
      // Cleanup
      document.body.classList.remove('hydrogeo-special-page');
      document.body.classList.remove('hydrogeo-dataset-page');
      document.body.classList.remove('hydrogeo-assistant-page');
      document.body.classList.remove('swatgenx-page');
    };
  }, [location.pathname]);

  // Add specific CSS classes to body for targeting SWATGenX page
  useEffect(() => {
    // Add "using-main-sidebar" class to body for SWATGenX page
    if (location.pathname === '/model_settings') {
      document.body.classList.add('using-main-sidebar');
    } else {
      document.body.classList.remove('using-main-sidebar');
    }
    
    return () => {
      document.body.classList.remove('using-main-sidebar');
    };
  }, [location.pathname]);

  return (
    <Box
      sx={{
        display: 'flex',
        bgcolor: '#2b2b2c',
        minHeight: '100vh',
        width: '100%',
        maxWidth: '100%',
        overflowX: 'hidden',
        color: 'white',
        position: 'relative',
      }}
    >
      {localStorage.getItem('authToken') && <SessionChecker />}

      {/* Only show the sidebar toggle button if not on the special pages */}
      {location.pathname !== '/hydrogeo-assistant' && 
       location.pathname !== '/hydro_geo_dataset' &&
       location.pathname !== '/model_settings' && (
        <IconButton
          onClick={(e) => {
            // Simple direct state update - clear and straightforward
            const newState = !isSidebarVisible;
            setIsSidebarVisible(newState);
            
            // Save state to session storage
            sessionStorage.setItem('hideSidebar', (!newState).toString());
            
            // Dispatch a direct, simple event
            window.dispatchEvent(new CustomEvent('sidebar-toggle-direct', { 
              detail: { visible: newState } 
            }));
            
            // Explicitly prevent event bubbling
            e.stopPropagation();
            e.preventDefault();
          }}
          aria-label="close-sidebar"
          id="sidebar-toggle-button"
          sx={{
            position: 'absolute',
            top: 16,
            left: isSidebarVisible ? (location.pathname === '/model_settings' ? drawerWidth + 16 : 16) : 16,
            zIndex: 1201,
            color: 'white',
            backgroundColor: 'rgba(43, 43, 44, 0.7)',
            '&:hover': {
              backgroundColor: 'rgba(43, 43, 44, 0.9)',
            },
          }}
        >
          {isSidebarVisible ? <CloseIcon /> : <MenuIcon />}
        </IconButton>
      )}

      {localStorage.getItem('authToken') && isSidebarVisible && (
        <Drawer
          variant="persistent"
          open={isSidebarVisible}
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
              // Added to ensure sidebar is properly positioned for fixed content pages
              position: 'fixed',
              height: '100vh',
              zIndex: 1200,
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

            <UserInfo userName={userName} />

            <List>
              {[
                { text: 'Home', icon: <HomeIcon />, path: '/' },
                { text: 'SWATGenX', icon: <SettingsIcon />, path: '/model_settings' },
                { text: 'Vision System', icon: <VisibilityIcon />, path: '/vision_system' },
                { text: 'HydroGeoDataset', icon: <StorageIcon />, path: '/hydro_geo_dataset' },
                { text: 'HydroGeo Assistant', icon: <SmartToyIcon />, path: '/hydrogeo-assistant' },
                { text: 'Dashboard', icon: <FolderIcon />, path: '/user_dashboard' },
                { text: 'Contact', icon: <EmailIcon />, path: '/contact' },
                { text: 'About', icon: <InfoIcon />, path: '/about' },
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
                      color: '#ff5722',
                    },
                  }}
                >
                  <ListItemIcon sx={{ color: 'inherit' }}>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              ))}
            </List>
          </Box>

          <Box sx={{ width: '100%' }}>
            <Divider sx={{ bgcolor: '#687891', mx: 2 }} />
            <List>
              <ListItemButton
                onClick={handleLogout}
                sx={{ color: 'white', '&:hover': { color: '#ff5722' } }}
              >
                <ListItemIcon>
                  <ExitToAppIcon sx={{ color: 'white' }} />
                </ListItemIcon>
                <ListItemText primary="Logout" />
              </ListItemButton>
            </List>
          </Box>

          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" sx={{ color: '#687891' }}>
              <Link to="/privacy" style={{ textDecoration: 'none', color: 'inherit' }}>
                Privacy Policy
              </Link>{' '}
              |
              <Link to="/terms" style={{ textDecoration: 'none', color: 'inherit', marginLeft: 8 }}>
                Terms of Service
              </Link>{' '}
              |
              <Link
                to="/ftps_server"
                style={{ textDecoration: 'none', color: 'inherit', marginLeft: 8 }}
              >
                FTPS Server
              </Link>
            </Typography>
          </Box>
        </Drawer>
      )}

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: (location.pathname === '/hydro_geo_dataset' || 
              location.pathname === '/hydrogeo-assistant' || 
              location.pathname === '/model_settings') ? 0 : 3, // No padding for special pages
          minHeight: '100vh',
          maxWidth: '100%',
          overflowX: 'hidden',
          marginLeft: ((location.pathname === '/hydro_geo_dataset' || 
                       location.pathname === '/hydrogeo-assistant' || 
                       location.pathname === '/model_settings') && isSidebarVisible) ? `${drawerWidth}px` : 0,
        }}
      >
        {location.pathname !== '/hydrogeo-assistant' && 
         location.pathname !== '/hydro_geo_dataset' && <Toolbar />}
        {children}
      </Box>
    </Box>
  );
};

export default Layout;
