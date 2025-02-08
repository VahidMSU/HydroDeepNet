// src/components/PrivateRoute.js
import React from 'react';
import { Navigate } from 'react-router-dom';

const PrivateRoute = ({ children }) => {
  const isAuthenticated = Boolean(localStorage.getItem('authToken'));
  return isAuthenticated ? children : <Navigate to="/login" replace />;
};

export default PrivateRoute;
