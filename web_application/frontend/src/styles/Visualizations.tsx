/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import { CircularProgress } from '@mui/material';
import colors from './colors.tsx';
import { Card, CardBody, ContentWrapper, ErrorMessage, LoadingContainer, SectionTitle } from './common.tsx';

// Layout components
export const Body = styled.div`
  padding: 2rem;
  background-color: ${colors.background};
  color: ${colors.text};
`;

export const PageTitle = styled.h1<{ $variant?: string }>`
  font-size: 2rem;
  margin-bottom: 2rem;
  color: ${colors.text};
`;

// Form related components
export const FormContainer = styled.form<{ $maxWidth?: string }>`
  padding: 2rem;
  background-color: ${colors.surface};
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  max-width: ${({ $maxWidth }) => $maxWidth || 'none'};
`;

export const FormGrid = styled.div`
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 2rem;
  margin-bottom: 2rem;
  flex-wrap: wrap;
`;

export const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

export const FormLabel = styled.label`
  font-weight: 600;
  color: ${colors.textSecondary};
  margin-bottom: 0.5rem;
`;

export const FormSelect = styled.select`
  padding: 0.75rem;
  border: 1px solid ${colors.border};
  border-radius: 4px;
  font-size: 1rem;
  background-color: ${colors.inputBg};
  color: ${colors.text};
  transition: border-color 0.2s;

  &:focus {
    border-color: ${colors.accent};
    outline: none;
    box-shadow: 0 0 0 2px ${colors.accent}33;
  }

  &[multiple] {
    height: 150px;
  }
`;

export const FormInput = styled.input`
  padding: 0.75rem;
  border: 1px solid ${colors.border};
  border-radius: 4px;
  font-size: 1rem;
  background-color: ${colors.inputBg};
  color: ${colors.text};
  transition: border-color 0.2s;

  &:focus {
    border-color: ${colors.accent};
    outline: none;
    box-shadow: 0 0 0 2px ${colors.accent}33;
  }
`;

export const HelpText = styled.small`
  font-size: 0.875rem;
  color: ${colors.textSecondary};
  margin-top: 0.25rem;
`;

export const SubmitButton = styled.button`
  background-color: ${colors.accent};
  color: ${colors.text};
  padding: 0.75rem 2rem;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background-color: ${colors.accentHover};
  }
`;

export const NoResults = styled.div`
  text-align: center;
  padding: 2rem;
  color: ${colors.textSecondary};
`;

// Collapsible components
export const Collapsible = styled.div`
  margin: 1rem 0;
  border: 1px solid ${colors.border};
  border-radius: 4px;
`;

export const HeaderTitle = styled.h1`
  text-align: center;
  margin-bottom: 30px;
  color: ${colors.text};
  font-size: 2.5rem;
  font-weight: bold;
`;

export const CollapsibleHeader = styled.div`
  padding: 1rem;
  cursor: pointer;
  background-color: ${colors.surfaceLight};
  color: ${colors.text};
  
  &:hover {
    background-color: ${colors.surface};
  }
`;

export const CollapsibleContent = styled.div`
  padding: 1rem;
  color: ${colors.text};
`;

export const CollapsibleText = styled.p`
  color: ${colors.text};
  line-height: 1.6;
  margin-bottom: 1rem;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

export const HeaderIcon = styled.span`
  margin-right: 8px;
`;

export const StyledCircularProgress = styled(CircularProgress)`
  color: ${colors.accent};
`;

// Content components
export const DescriptionContainer = styled.div`
  background-color: #444e5e;
  padding: 24px;
  margin: 20px 0;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  color: white;
  margin-bottom: 20px;
`;

export const Description = styled.p`
  margin-bottom: 1rem;
  line-height: 1.6;
  color: ${colors.text};
  
  &:last-child {
    margin-bottom: 0;
  }
`;

export const VisualizationSection = styled.section`
  // Additional styling if needed
`;

export const GifContainer = styled.div`
  display: grid;
  gap: 2rem;
  margin-top: 2rem;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
`;

export const GifWrapper = styled.div`
  border-radius: 12px;
  overflow: hidden;
  background-color: rgba(255, 255, 255, 0.1);
  transition: transform 0.3s ease;

  &:hover {
    transform: translateY(-5px);
  }

  img {
    width: 100%;
    height: auto;
    object-fit: cover;
  }
`;

export const DownloadButton = styled.a`
  color: #ff5722;
  text-decoration: none;
  padding: 0.5rem 1rem;
  border: 2px solid #ff5722;
  border-radius: 8px;
  margin-top: 1rem;
  display: inline-block;
  transition: all 0.3s ease;

  &:hover {
    background-color: #ff5722;
    color: #ffffff;
  }
`;

export const VisualizationContainer = styled.div`
  max-width: 1200px;
  margin: 2rem auto;
  padding: 2rem;
  color: #ffffff;
`;

export const VisualizationTitle = styled.h2`
  color: #ff5722;
  font-size: 2.8rem;
  margin-bottom: 1.5rem;
  border-bottom: 3px solid #ff5722;
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

// Export existing components from common
export { Card, CardBody, ContentWrapper, ErrorMessage, LoadingContainer, SectionTitle };
