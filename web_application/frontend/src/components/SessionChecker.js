import React, { useEffect, useState } from 'react';
import styled from '@emotion/styled';
import { useLocation, useNavigate } from 'react-router-dom';
import SessionService from '../services/SessionService';

const SessionWarning = styled.div`
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 15px 20px;
  background-color: #2b2b2c;
  color: #ffd380;
  border: 1px solid #ff5722;
  border-radius: 6px;
  z-index: 9999;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  display: flex;
  flex-direction: column;
  align-items: flex-start;
`;

const WarningText = styled.p`
  margin: 0 0 10px 0;
  font-size: 14px;
  line-height: 1.5;
`;

const ButtonContainer = styled.div`
  display: flex;
  gap: 10px;
  align-self: flex-end;
`;

const Button = styled.button`
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 600;
  font-size: 13px;
  transition: background-color 0.2s;
  
  &.primary {
    background-color: #ff5722;
    color: #fff;
    
    &:hover {
      background-color: #e67700;
    }
  }
  
  &.secondary {
    background-color: transparent;
    color: #ccc;
    border: 1px solid #444;
    
    &:hover {
      background-color: #333;
    }
  }
`;

/**
 * Component to initialize and maintain session checks
 * This component doesn't render anything
 */
const SessionChecker = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    // Check if this is a Google OAuth redirect
    const checkOAuthRedirect = () => {
      const urlParams = new URLSearchParams(location.search);
      const isOAuthRedirect = urlParams.has('google_login') && urlParams.has('username');
      
      if (isOAuthRedirect) {
        console.log('SessionChecker detected OAuth redirect, processing...');
        SessionService.checkGoogleOAuthLogin(window.location.href);
      }
    };
    
    // First check for OAuth redirect
    checkOAuthRedirect();
    
    // Then start session monitoring
    SessionService.startSessionMonitor();

    // Perform an immediate session check
    SessionService.checkSession();

    // Clean up when component unmounts
    return () => {
      SessionService.stopSessionMonitor();
    };
  }, [location.search]);

  // This component doesn't render anything
  return null;
};

export default SessionChecker;
