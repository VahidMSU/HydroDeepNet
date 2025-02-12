/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import { Box, Button } from '@mui/material';

// Dark theme colors
const colors = {
  background: '#2b2b2c',
  surface: '#444e5e',
  accent: '#ff8500',
  accentHover: '#ffa533',
  text: '#ffffff',
  border: '#ff8500',
  disabled: '#666666',
  error: '#ff4444',
};

// Base styles
const baseInputStyles = `
  background-color: ${colors.background};
  border: 2px solid ${colors.accent};
  border-radius: 6px;
  color: ${colors.text};
  font-family: 'Roboto', sans-serif;
  font-size: 1rem;
  padding: 0.8rem;
  transition: all 0.2s ease-in-out;

  &:focus {
    outline: none;
    border-color: ${colors.accentHover};
    box-shadow: 0 0 0 2px rgba(255, 133, 0, 0.2);
  }

  &:hover {
    border-color: ${colors.accentHover};
  }
`;

// Styled components
export const PageLayout = styled.div`
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 20px;
  height: 100vh;
  padding: 20px;
  background-color: ${colors.background};
`;

// Enhanced form field styles
export const StyledForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  padding: 2rem;
  background-color: ${colors.surface};
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 133, 0, 0.1);
`;

export const FormSection = styled.div`
  position: relative;
  margin-bottom: 2rem;
  padding: 1.5rem;
  background-color: ${colors.background};
  border-radius: 8px;
  border: 1px solid ${colors.border};
  transition: all 0.2s ease-in-out;

  &:hover {
    box-shadow: 0 4px 12px rgba(255, 133, 0, 0.1);
  }
`;

export const SectionTitle = styled.h3`
  color: ${colors.accent};
  font-size: 1.3rem;
  font-weight: 500;
  margin-bottom: 1.5rem;
  padding-bottom: 0.8rem;
  border-bottom: 2px solid ${colors.accent};
  display: flex;
  align-items: center;
  gap: 0.8rem;

  .icon {
    font-size: 1.2rem;
  }
`;

export const FormFieldGroup = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 1.5rem;
`;

export const FormField = styled.div`
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

export const Label = styled.label`
  color: ${colors.accent};
  font-size: 0.95rem;
  font-weight: 500;
  margin-bottom: 0.3rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;

  .icon {
    font-size: 1rem;
    opacity: 0.8;
  }
`;

export const InputWrapper = styled.div`
  position: relative;
  width: 100%;
`;

export const Input = styled.input`
  width: 100%;
  padding: 0.9rem 1rem;
  background-color: ${colors.background};
  border: 2px solid ${colors.border};
  border-radius: 6px;
  color: ${colors.text};
  font-size: 1rem;
  transition: all 0.2s ease-in-out;

  &:focus {
    outline: none;
    border-color: ${colors.accentHover};
    box-shadow: 0 0 0 3px rgba(255, 133, 0, 0.2);
  }

  &:hover:not(:disabled) {
    border-color: ${colors.accentHover};
  }

  &:disabled {
    background-color: ${colors.disabled};
    cursor: not-allowed;
    opacity: 0.7;
  }

  &::placeholder {
    color: rgba(255, 255, 255, 0.4);
  }
`;

export const Select = styled.select`
  width: 100%;
  padding: 0.9rem 1rem;
  background-color: ${colors.background};
  border: 2px solid ${colors.border};
  border-radius: 6px;
  color: ${colors.text};
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  appearance: none;
  background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ff8500'%3e%3cpath d='M7 10l5 5 5-5z'/%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right 1rem center;
  background-size: 1.5rem;

  &:focus {
    outline: none;
    border-color: ${colors.accentHover};
    box-shadow: 0 0 0 3px rgba(255, 133, 0, 0.2);
  }

  &:hover:not(:disabled) {
    border-color: ${colors.accentHover};
  }

  option {
    background-color: ${colors.background};
    color: ${colors.text};
    padding: 0.8rem;
  }
`;

export const CheckboxGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 1.2rem;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
`;

export const CheckboxWrapper = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem;
  border-radius: 6px;
  transition: all 0.2s ease-in-out;

  &:hover {
    background-color: rgba(255, 133, 0, 0.1);
  }

  input[type="checkbox"] {
    width: 20px;
    height: 20px;
    accent-color: ${colors.accent};
    cursor: pointer;

    &:checked {
      & + label {
        color: ${colors.accent};
      }
    }
  }

  label {
    color: ${colors.text};
    cursor: pointer;
    user-select: none;
    font-size: 0.95rem;
  }
`;

export const StationInfoCard = styled.div`
  background-color: ${colors.background};
  border: 1px solid ${colors.border};
  border-radius: 8px;
  padding: 1.2rem;
  margin-top: 1rem;

  h4 {
    color: ${colors.accent};
    font-size: 1.1rem;
    margin-bottom: 1rem;
  }

  p {
    color: ${colors.text};
    margin: 0.5rem 0;
    font-size: 0.9rem;
    display: flex;
    justify-content: space-between;
  }

  span.label {
    color: ${colors.accent};
    font-weight: 500;
  }
`;

export const SubmitButton = styled(Button)`
  background-color: ${colors.accent};
  color: ${colors.text};
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  font-weight: 500;
  border-radius: 6px;
  text-transform: none;
  margin-top: 1rem;
  transition: all 0.2s ease-in-out;

  &:hover {
    background-color: ${colors.accentHover};
    transform: translateY(-1px);
  }

  &:disabled {
    background-color: ${colors.disabled};
    color: rgba(255, 255, 255, 0.5);
  }
`;

export const ErrorMessage = styled.div`
  color: ${colors.error};
  background-color: rgba(255, 68, 68, 0.1);
  border: 1px solid ${colors.error};
  border-radius: 6px;
  padding: 0.8rem;
  margin: 1rem 0;
  font-size: 0.9rem;
`;

const ContainerFluid = styled.div`
  padding: 15px 25px 15px 15px;  // Added right padding
  background-color: #f8f9fa;
  min-height: 100vh;
  max-width: 100vw;
`;

export const ContentWrapper = styled.div`
  flex-grow: 1;
  margin-left: 270px;
  padding: 2.5rem;
  background-color: #ffffff;
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
  overflow-y: auto;
  height: calc(100vh - 220px);
  transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);

  &:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
  }

  @media (max-width: 768px) {
    margin-left: 0;
    height: auto;
    max-height: 700px;
  }
`;

export const FooterContainer = styled.footer`
  width: 10%;
  margin-left: 0;
  background-color: #343a40;
  color: #343a40;
  text-align: center;
  padding: 15px;
  font-size: 10px;
  border-top: 1px solid #343a40;
  position: fixed;
  bottom: 0;

  a {
    color: #d8dbde;
    text-decoration: none;
    margin: 0 10px;
    font-weight: 500;

    &:hover {
      text-decoration: underline;
    }
  }
`;

const Title = styled.h2`
  text-align: center;
  margin: 40px 0;
  color: #333;
`;

const Row = styled.div`
  display: flex;
  flex-wrap: wrap;
  width: 100%;
  height: 100%;
  gap: 1.5rem;  // Increased gap between columns
  margin: 0;
  padding: 0;  // Removed small horizontal padding
`;

export const Content = styled.main`
  margin-left: 65px;  // Reduced margin to move closer to sidebar
  padding: 0 20px 0 0;  // Added right padding
  display: flex;
  flex-wrap: wrap;
  width: calc(100vw - 85px);  // Adjusted for new spacing
  height: calc(100vh - 30px); // Minimal spacing for header
  overflow: hidden;
  box-sizing: border-box;

  @media (max-width: 768px) {
    margin-left: 0;
    width: 100vw;
    padding: 0 10px;
  }
`;

interface ColumnProps {
  width?: number;
  minWidth?: string;
  mobileMinWidth?: string;
}

const Column = styled.div<ColumnProps>`
  flex: ${props => props.width || 1};
  min-width: ${props => props.minWidth || '400px'};
  height: 100%;
  display: flex;
  flex-direction: column;

  &.map-column {
    position: relative;
    overflow: hidden;
  }

  @media (max-width: 1400px) {
    min-width: ${props => props.mobileMinWidth || '100%'};
  }
`;

export const MapWrapper = styled(ContentWrapper)`
  height: 100% !important;
  width: 100%;
  margin: 0;
  padding: 1rem 1.5rem 1rem 1rem;  // Increased right padding
  position: sticky;
  top: 0;
  overflow: hidden;
  transform: none !important;

  &:hover {
    transform: none !important;
  }

  @media (max-width: 1400px) {
    height: 80vh !important;
  }
`;

const Card = styled.div`
  background-color: #ffffff;
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
  overflow: hidden;
  margin-bottom: 2rem;
  border: 1px solid #eef0f2;
  transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);

  &:hover {
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
    border-color: #e0e4e8;
  }
`;

const CardBody = styled.div`
  padding: 0.75rem;  // Adjusted padding
`;

export const DescriptionContainer = styled.div`
  background-color: #444e5e;
  padding: 24px;
  margin: 20px 0;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  color: white;

  p {
    margin-bottom: 1rem;
    line-height: 1.6;
    
    &:last-child {
      margin-bottom: 0;
    }
  }

  ul {
    margin: 1rem 0;
    padding-left: 1.5rem;
  }

  li {
    margin-bottom: 0.5rem;
  }

  strong {
    color: #ff8500;
    font-weight: 500;
  }

  margin-bottom: 20px;
  
  .description-header:hover {
    background-color: #e9e9e9 !important;
  }
`;

export const InfoBox = styled.div`
  background-color: #2b2b2c;
  border: 1px solid #ff8500;
  border-radius: 8px;
  padding: 1.5rem;
  margin-top: 1rem;

  p {
    color: white;
    margin-bottom: 1rem;
  }

  ul {
    color: white;
    list-style-type: disc;
    margin-left: 1.5rem;
  }

  li {
    margin-bottom: 0.5rem;
  }

  strong {
    color: #ff8500;
  }
`;

export const FormContainer = styled(Box)({
  padding: '20px',
  borderRadius: '12px',
  backgroundColor: '#2b2b2c',
  color: 'white',
  '& .MuiInputBase-root': {
    color: 'white',
  },
  '& .MuiFormLabel-root': {
    color: '#ff8500',
  },
  '& .MuiOutlinedInput-root': {
    '& fieldset': {
      borderColor: '#ff8500',
    },
    '&:hover fieldset': {
      borderColor: '#ff8500',
    },
  },
});

export const FormGroupLabel = styled('label')({
  display: 'block',
  marginBottom: '5px',
  fontWeight: 'bold',
  textAlign: 'match-parent',
});

export { ContainerFluid, Title, Row, Column, Card, CardBody };
