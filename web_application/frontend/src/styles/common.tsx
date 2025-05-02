/** @jsxImportSource @emotion/react */
import { keyframes } from '@emotion/react';
import styled from '@emotion/styled';
import colors from './colors.tsx';
import { css } from '@emotion/react';

// Common animations that can be reused across components
export const fadeIn = keyframes`
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

export const slideDown = keyframes`
  from {
    opacity: 0;
    transform: translateY(-10px);
    max-height: 0;
  }
  to {
    opacity: 1;
    transform: translateY(0);
    max-height: 250px;
  }
`;

export const spin = keyframes`
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
`;

// Common UI elements shared across multiple components
export const ContentWrapper = styled.div`
  background: linear-gradient(145deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 16px;
  padding: 2.2rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  border: 1px solid ${colors.border};
  overflow: hidden;
  position: relative;
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(to right, ${colors.accent}80, transparent);
  }
`;

export const SectionTitle = styled.h2`
  font-size: 1.5rem;
  margin: 2rem 0;
  color: ${colors.text};
`;

export const Card = styled.div`
  background-color: ${colors.surface};
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  padding: 1rem;
  margin: 1rem 0;
  border: 1px solid ${colors.border};
  color: ${colors.text};
`;

export const CardBody = styled.div`
  padding: 1rem;
`;

export const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  padding: 2rem;
`;

export const ErrorMessage = styled.div`
  color: ${colors.accentAlt};
  background-color: ${colors.surfaceLight};
  padding: 1rem;
  border-radius: 4px;
  margin-top: 1rem;
  text-align: center;
`;

export const SectionHeader = styled.h3`
  color: #ff5722;
  font-size: 1.8rem;
  margin: 2rem 0 1rem;
`;

export const SubmitButton = styled.button`
  background-color: ${colors.accent};
  color: ${colors.text};
  padding: 0.75rem 2rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    background-color: ${colors.accentHover};
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(255, 133, 0, 0.3);
  }
  
  &:active {
    transform: translateY(0);
  }
`;

// Add common title components used across the application
export const PageTitle = styled.h1`
  color: ${colors.accent};
  font-size: 2.8rem;
  margin-bottom: 1.5rem;
  border-bottom: 3px solid ${colors.accent};
  padding-bottom: 1.2rem;
  position: relative;
  text-align: center;
  
  &:after {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background-color: ${colors.accentHover};
  }
`;

// Add title components used in multiple files (TermsConditions, Privacy, Visualizations, Michigan)
export const TitleBase = styled.h2`
  color: ${colors.accent};
  font-size: 2.8rem;
  margin-bottom: 1.8rem;
  border-bottom: 3px solid ${colors.accent};
  padding-bottom: 1.2rem;
  position: relative;
  text-align: center;
  
  &:after {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, ${colors.accent}, ${colors.accentHover});
    border-radius: 3px;
  }
`;

// Unified container for main sections to avoid duplication
export const SectionContainer = styled.div`
  max-width: 1200px;
  margin: 2rem auto;
  padding: 2rem;
  color: ${colors.text};
  background-color: ${colors.background};
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
`;

// Unified link styling
export const StyledLinks = styled.div`
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 1rem;

  a {
    color: ${colors.accent};
    text-decoration: none;
    padding: 0.5rem 1rem;
    border: 2px solid ${colors.accent};
    border-radius: 8px;
    transition: all 0.3s ease;
    background: linear-gradient(to right, transparent 50%, ${colors.accent} 50%);
    background-size: 200% 100%;
    background-position: 0 0;

    &:hover {
      background-position: -100% 0;
      color: ${colors.textInverse};
    }
  }
`;

// Common form elements
export const FormContainer = styled.form`
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
  background: linear-gradient(145deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
  border: 1px solid ${colors.border};
`;

export const FormGroup = styled.div`
  margin-bottom: 1.5rem;
`;

export const FormLabel = styled.label`
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: ${colors.textSecondary};
  position: relative;
  padding-left: 0.5rem;
  
  &:before {
    content: '';
    position: absolute;
    left: 0;
    top: 25%;
    height: 50%;
    width: 3px;
    background: ${colors.accent};
    border-radius: 2px;
  }
`;

export const FormInput = styled.input`
  width: 100%;
  padding: 0.75rem;
  border: 1px solid ${colors.border};
  border-radius: 6px;
  background-color: ${colors.inputBg};
  color: ${colors.inputText};
  transition: all 0.2s ease;
  
  &:focus {
    outline: none;
    border-color: ${colors.accent};
    box-shadow: 0 0 0 3px ${colors.accent}25;
  }
  
  &:hover:not(:focus) {
    border-color: ${colors.borderLight};
  }
`;

// Common modal components
export const Modal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

export const ModalContent = styled.div`
  background-color: ${colors.surface};
  border-radius: 8px;
  padding: 1.5rem;
  position: relative;
  max-width: 90%;
  max-height: 90%;
`;

export const CloseButton = styled.button`
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: transparent;
  border: none;
  color: ${colors.textSecondary};
  font-size: 1.5rem;
  cursor: pointer;
  transition: color 0.2s ease;
  
  &:hover {
    color: ${colors.accent};
  }
`;

// Add a consistent background color for all pages matching the layout background
export const globalStyles = css`
  body, html {
    background-color: ${colors.background};
    color: ${colors.text};
  }
  
  .MuiBox-root {
    background-color: ${colors.background};
  }
  
  // Make sure all components with background inherit this color
  .map-container, 
  .report-container, 
  .form-container, 
  .station-details-container,
  .info-box-container,
  .search-form-container,
  .user-info-container,
  .viewer-container {
    background-color: ${colors.surface};
  }
`;

// Add or update any component containers that need the background color
export const PageContainer = styled.div`
  background-color: ${colors.background};
  min-height: 100vh;
  display: flex;
  flex-direction: column;
`;

export const ContentContainer = styled.div`
  background-color: ${colors.background};
  flex: 1;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
`;

export const ComponentContainer = styled.div`
  background: linear-gradient(145deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 1.5rem;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
  border: 1px solid ${colors.border};
  
  &:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  }
`;
