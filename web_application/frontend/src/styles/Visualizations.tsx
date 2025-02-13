import styled from 'styled-components';
import { CircularProgress } from '@mui/material';

// Define our color palette
const colors = {
  background: '#2b2b2c',    // Dark background
  surface: '#2b2b2c',       // Surface for containers/cards
  surfaceLight: '#2b2b2c',
  accent: '#ff8500',        // Bright accent color
  accentHover: '#00202e',
  accentAlt: '#ff6361',
  text: '#ffd380',          // Main text color (lighter grey/white)
  textSecondary: '#bbbbbb',
  textMuted: '#ffd380',
  border: '#687891',
  borderAccent: '#00202e',
  inputBg: '#bbbbbb',
  headerBg: '#2b2b2c',
  TitleText: '#00202e',
  SectionText: '#00202e',
  FieldText: 'white'
};

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
  background-color: #ffffff;
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
  color: #333;
  margin-bottom: 0.5rem;
`;

export const FormSelect = styled.select`
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  background-color: #fff;
  transition: border-color 0.2s;

  &:focus {
    border-color: #2196f3;
    outline: none;
    box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1);
  }

  &[multiple] {
    height: 150px;
  }
`;

export const FormInput = styled.input`
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  background-color: #fff;
  transition: border-color 0.2s;

  &:focus {
    border-color: #2196f3;
    outline: none;
    box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1);
  }
`;

export const HelpText = styled.small`
  font-size: 0.875rem;
  color: #666;
  margin-top: 0.25rem;
`;

export const SubmitButton = styled.button`
  background-color: #2196f3;
  color: white;
  padding: 0.75rem 2rem;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background-color: #1976d2;
  }
`;

export const ErrorMessage = styled.div`
  color: #d32f2f;
  background-color: #ffebee;
  padding: 1rem;
  border-radius: 4px;
  margin-top: 1rem;
  text-align: center;
`;

// Additional required exports
export const SectionTitle = styled.h2<{ $variant?: string }>`
  font-size: 1.5rem;
  margin: 2rem 0;
  color: ${colors.text};
`;

export const GifContainer = styled.div`
  display: grid;
  gap: 2rem;
  margin-top: 2rem;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));

  @media (min-width: 1200px) {
    grid-template-columns: repeat(4, 1fr); // maximum 4 columns on wider screens
  }
`;

export const GifWrapper = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;

  img {
    width: 100%;
    height: auto;
    object-fit: contain;
  }
`;

export const DownloadButton = styled.a`
  background-color: #4caf50;
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  text-decoration: none;
  
  &:hover {
    background-color: #388e3c;
  }
`;

export const NoResults = styled.div`
  text-align: center;
  padding: 2rem;
  color: #666;
`;

// Collapsible components
export const Collapsible = styled.div`
  margin: 1rem 0;
  border: 1px solid #ddd;
  border-radius: 4px;
`;

export const CollapsibleHeader = styled.div`
  padding: 1rem;
  cursor: pointer;
  background-color: #f5f5f5;
  
  &:hover {
    background-color: #eeeeee;
  }
`;

export const CollapsibleContent = styled.div`
  padding: 1rem;
`;

export const CollapsibleText = styled.p`
  color: #333;
  line-height: 1.6;
  margin-bottom: 1rem;
`;

export const HeaderIcon = styled.span`
  margin-right: 8px;
`;

// Loading components
export const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  padding: 2rem;
`;

export const StyledCircularProgress = styled(CircularProgress)`
  color: #2196f3;
`;

// Content components
export const DescriptionContainer = styled.div`
  margin: 2rem 0;
  line-height: 1.6;
  
  p {
    margin-bottom: 1rem;
    
    &:last-child {
      margin-bottom: 0;
    }
  }
`;

export const Description = styled.p`
  margin-bottom: 1rem;
  line-height: 1.6;
  color: #333;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

export const VisualizationSection = styled.section`
  // Additional styling if needed
`;

// New Card components for styling improvements
export const Card = styled.div`
  background-color: ${colors.surface};
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  padding: 1rem;
  margin: 1rem 0;
  border: 1px solid ${colors.border};
`;

export const CardBody = styled.div`
  padding: 1rem;
`;
