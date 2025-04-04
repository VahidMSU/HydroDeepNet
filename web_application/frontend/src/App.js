import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
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

const App = () => {
  return (
    <Router>
      <Suspense fallback={<LoadingFallback />}>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/verify" element={<Verify />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/reset-password" element={<ResetPassword />} />
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
