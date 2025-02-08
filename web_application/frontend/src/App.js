import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './pages/Layout';
import {
  Home,
  ModelSettings,
  Visualizations,
  Michigan,
  VisionSystem,
  HydroGeoDataset,
  UserDashboard,
  Logout,
  Contact,
  About,
  SignUp,
  Privacy,
  Terms,
  Login,
} from './pages';
import PrivateRoute from './components/PrivateRoute';

const App = () => (
  <Router>
    <Routes>
      <Route
        path="/login"
        element={
          <Layout>
            <Login />
          </Layout>
        }
      />
      <Route
        path="/"
        element={
          <PrivateRoute>
            <Layout>
              <Home />
            </Layout>
          </PrivateRoute>
        }
      />
      <Route
        path="/model_settings"
        element={
          <PrivateRoute>
            <Layout>
              <ModelSettings />
            </Layout>
          </PrivateRoute>
        }
      />
      <Route
        path="/visualizations"
        element={
          <PrivateRoute>
            <Layout>
              <Visualizations />
            </Layout>
          </PrivateRoute>
        }
      />
      <Route
        path="/michigan"
        element={
          <PrivateRoute>
            <Layout>
              <Michigan />
            </Layout>
          </PrivateRoute>
        }
      />
      <Route
        path="/vision_system"
        element={
          <PrivateRoute>
            <Layout>
              <VisionSystem />
            </Layout>
          </PrivateRoute>
        }
      />
      <Route
        path="/hydro_geo_dataset"
        element={
          <PrivateRoute>
            <Layout>
              <HydroGeoDataset />
            </Layout>
          </PrivateRoute>
        }
      />
      <Route
        path="/user_dashboard"
        element={
          <PrivateRoute>
            <Layout>
              <UserDashboard />
            </Layout>
          </PrivateRoute>
        }
      />
      <Route
        path="/api/logout"
        element={
          <PrivateRoute>
            <Layout>
              <Logout />
            </Layout>
          </PrivateRoute>
        }
      />
      <Route
        path="/contact"
        element={
          <Layout>
            <Contact />
          </Layout>
        }
      />
      <Route
        path="/about"
        element={
          <Layout>
            <About />
          </Layout>
        }
      />
      <Route
        path="/signup"
        element={
          <Layout>
            <SignUp />
          </Layout>
        }
      />
      <Route
        path="/privacy"
        element={
          <Layout>
            <Privacy />
          </Layout>
        }
      />
      <Route
        path="/terms"
        element={
          <Layout>
            <Terms />
          </Layout>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </Router>
);

export default App;
