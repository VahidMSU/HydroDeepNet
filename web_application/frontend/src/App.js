import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './pages/Layout';
import {
  Home,
  SWATGenX,
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

const PRIVATE_MODE = process.env.REACT_APP_PRIVATE_MODE === 'true';

const App = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={PRIVATE_MODE ? <PrivateRoute><Home /></PrivateRoute> : <Home />} />
          <Route path="/model_settings" element={PRIVATE_MODE ? <PrivateRoute><SWATGenX /></PrivateRoute> : <SWATGenX />} />
          <Route path="/visualizations" element={PRIVATE_MODE ? <PrivateRoute><Visualizations /></PrivateRoute> : <Visualizations />} />
          <Route path="/michigan" element={PRIVATE_MODE ? <PrivateRoute><Michigan /></PrivateRoute> : <Michigan />} />
          <Route path="/vision_system" element={PRIVATE_MODE ? <PrivateRoute><VisionSystem /></PrivateRoute> : <VisionSystem />} />
          <Route path="/hydro_geo_dataset" element={PRIVATE_MODE ? <PrivateRoute><HydroGeoDataset /></PrivateRoute> : <HydroGeoDataset />} />
          <Route path="/user_dashboard" element={PRIVATE_MODE ? <PrivateRoute><UserDashboard /></PrivateRoute> : <UserDashboard />} />
          <Route path="/logout" element={PRIVATE_MODE ? <PrivateRoute><Logout /></PrivateRoute> : <Logout />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/about" element={<About />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="/terms" element={<Terms />} />
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </Layout>
    </Router>
  );
};

export default App;