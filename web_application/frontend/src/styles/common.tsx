/** @jsxImportSource @emotion/react */
import { keyframes } from '@emotion/react';
import styled from '@emotion/styled';
import colors from './colors.tsx';

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
  background-color: #444e5e;
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
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
  color: #ff8500;
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
  color: #ff8500;
  font-size: 2.8rem;
  margin-bottom: 1.5rem;
  border-bottom: 3px solid #ff8500;
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
    background-color: #ffa533;
  }
`;

// Unified container for main sections to avoid duplication
export const SectionContainer = styled.div`
  max-width: 1200px;
  margin: 2rem auto;
  padding: 2rem;
  color: #ffffff;
`;

// Unified link styling
export const StyledLinks = styled.div`
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 1rem;

  a {
    color: #ff8500;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border: 2px solid #ff8500;
    border-radius: 8px;
    transition: all 0.3s ease;

    &:hover {
      background-color: #ff8500;
      color: #ffffff;
    }
  }
`;

// Common form elements
export const FormContainer = styled.form`
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
`;

export const FormGroup = styled.div`
  margin-bottom: 1.5rem;
`;

export const FormLabel = styled.label`
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: ${colors.textSecondary};
`;

export const FormInput = styled.input`
  width: 100%;
  padding: 0.75rem;
  border: 1px solid ${colors.border};
  border-radius: 4px;
  background-color: ${colors.inputBg};
  color: ${colors.inputText};
  transition: all 0.2s ease;
  
  &:focus {
    outline: none;
    border-color: ${colors.accent};
    box-shadow: 0 0 0 2px rgba(255, 133, 0, 0.15);
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
