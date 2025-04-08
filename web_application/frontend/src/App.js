import React, { Suspense, lazy, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import Layout from './pages/Layout';
import PrivateRoute from './components/PrivateRoute';

const PRIVATE_MODE = process.env.REACT_APP_PRIVATE_MODE === 'true';

// Replace synchronous imports with lazy imports
const Home = lazy(() => import('./pages/Home'));
const SWATGenX = lazy(() => import('./pages/SWATGenX'));
const Visualizations = lazy(() => import('./pages/Visualizations'));
const Michigan = lazy(() => import('./pages/Michigan'));
const ForgotPassword = lazy(() => import('./pages/ForgotPassword'));
const ResetPassword = lazy(() => import('./pages/ResetPassword'));
const VisionSystem = lazy(() => import('./pages/VisionSystem'));
const HydroGeoDataset = lazy(() => import('./pages/HydroGeoDataset'));
const HydroGeoAssistant = lazy(() => import('./pages/HydroGeoAssistant'));
const UserDashboard = lazy(() => import('./pages/UserDashboard'));
const ContactUs = lazy(() => import('./pages/ContactUs'));
const AboutUs = lazy(() => import('./pages/AboutUs'));
const Login = lazy(() => import('./pages/Login'));
const Privacy = lazy(() => import('./pages/Privacy'));
const Terms = lazy(() => import('./pages/Terms'));
const SignUp = lazy(() => import('./pages/SignUp'));
const Verify = lazy(() => import('./pages/Verify'));
const Logout = lazy(() => import('./pages/Logout'));
const FTPSServer = lazy(() => import('./pages/FTPSServer'));

// Loading fallback component
const LoadingFallback = () => (
  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
    <div className="spinner"></div>
  </div>
);

// Scroll state cleanup component
const ScrollStateCleanup = () => {
  const location = useLocation();
  
  useEffect(() => {
    // If we're navigating to a regular page (not a no-scroll page)
    if (
      !location.pathname.includes('hydrogeo-assistant') && 
      !location.pathname.includes('hydro_geo_dataset') && 
      !location.pathname.includes('model_settings')
    ) {
      // Check if we need to clean up scroll classes
      const cameFromNoScroll = sessionStorage.getItem('came_from_noscroll') === 'true';
      
      if (cameFromNoScroll) {
        console.log('Cleaning up no-scroll state');
        // Clean up classes
        document.documentElement.classList.remove('no-scroll-page');
        document.body.classList.remove('no-scroll-page');
        
        // Reset the flag
        sessionStorage.removeItem('came_from_noscroll');
      }
    }
  }, [location.pathname]);
  
  return null;
};

const App = () => {
  return (
    <Router>
      <ScrollStateCleanup />
      <Suspense fallback={<LoadingFallback />}>
        <Routes>
          {/* Public routes without Layout */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/verify" element={<Verify />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/reset-password" element={<ResetPassword />} />

          {/* SWATGenX route - standalone with no Layout/sidebar */}
          <Route
            path="/model_settings"
            element={
              PRIVATE_MODE ? (
                <PrivateRoute>
                  <SWATGenX />
                </PrivateRoute>
              ) : (
                <SWATGenX />
              )
            }
          />

          {/* Private routes with Layout */}
          <Route
            path="/*"
            element={
              <Layout>
                <Routes>
                  <Route
                    path="/"
                    element={
                      PRIVATE_MODE ? (
                        <PrivateRoute>
                          <Home />
                        </PrivateRoute>
                      ) : (
                        <Home />
                      )
                    }
                  />
                  <Route
                    path="/visualizations"
                    element={
                      PRIVATE_MODE ? (
                        <PrivateRoute>
                          <Visualizations />
                        </PrivateRoute>
                      ) : (
                        <Visualizations />
                      )
                    }
                  />
                  <Route
                    path="/michigan"
                    element={
                      PRIVATE_MODE ? (
                        <PrivateRoute>
                          <Michigan />
                        </PrivateRoute>
                      ) : (
                        <Michigan />
                      )
                    }
                  />
                  <Route
                    path="/vision_system"
                    element={
                      PRIVATE_MODE ? (
                        <PrivateRoute>
                          <VisionSystem />
                        </PrivateRoute>
                      ) : (
                        <VisionSystem />
                      )
                    }
                  />
                  <Route
                    path="/hydro_geo_dataset"
                    element={
                      PRIVATE_MODE ? (
                        <PrivateRoute>
                          <HydroGeoDataset />
                        </PrivateRoute>
                      ) : (
                        <HydroGeoDataset />
                      )
                    }
                  />
                  <Route
                    path="/hydrogeo-assistant"
                    element={
                      PRIVATE_MODE ? (
                        <PrivateRoute>
                          <HydroGeoAssistant />
                        </PrivateRoute>
                      ) : (
                        <HydroGeoAssistant />
                      )
                    }
                  />
                  <Route
                    path="/user_dashboard"
                    element={
                      PRIVATE_MODE ? (
                        <PrivateRoute>
                          <UserDashboard />
                        </PrivateRoute>
                      ) : (
                        <UserDashboard />
                      )
                    }
                  />
                  <Route
                    path="/logout"
                    element={
                      PRIVATE_MODE ? (
                        <PrivateRoute>
                          <Logout />
                        </PrivateRoute>
                      ) : (
                        <Logout />
                      )
                    }
                  />
                  <Route path="/contact" element={<ContactUs />} />
                  <Route path="/about" element={<AboutUs />} />
                  <Route path="/privacy" element={<Privacy />} />
                  <Route path="/terms" element={<Terms />} />
                  <Route
                    path="/ftps_server"
                    element={
                      PRIVATE_MODE ? (
                        <PrivateRoute>
                          <FTPSServer />
                        </PrivateRoute>
                      ) : (
                        <FTPSServer />
                      )
                    }
                  />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Layout>
            }
          />
        </Routes>
      </Suspense>
    </Router>
  );
};

export default App;
