import styled from 'styled-components';
import { Container, Typography, Button, CircularProgress } from '@mui/material';

export const Body = styled.div`
  background-color: #2b2b2c;
  min-height: 100vh;
  padding: 20px;
`;

export const FormContainer = styled(Container)`
  background-color: #444e5e;
  padding: 24px;
  border-radius: 12px;
  margin-top: 20px;
  margin-bottom: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
`;

export const PageTitle = styled(Typography)`
  color: #ff8500;
  font-family: 'Roboto', sans-serif;
  font-size: 2.5rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 2rem;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
`;

export const SectionTitle = styled(Typography)`
  color: #ff8500;
  font-family: 'Roboto', sans-serif;
  font-size: 1.8rem;
  font-weight: 600;
  margin: 2rem 0 1rem;
  border-bottom: 2px solid #ff8500;
  padding-bottom: 0.5rem;
`;

export const GifContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 20px;
`;

export const GifWrapper = styled.div`
  background-color: #2b2b2c;
  padding: 16px;
  border-radius: 8px;
  border: 1px solid #ff8500;
  
  img {
    width: 100%;
    border-radius: 4px;
  }
`;

export const NoResults = styled.div`
  color: white;
  text-align: center;
  padding: 24px;
  background-color: #2b2b2c;
  border-radius: 8px;
  border: 1px solid #ff8500;
`;

export const DescriptionContainer = styled.div`
  color: white;
  margin-bottom: 24px;
  padding: 16px;
  background-color: #2b2b2c;
  border-radius: 8px;
  border: 1px solid #ff8500;
`;

export const Collapsible = styled.div`
  margin-bottom: 16px;
  border: 1px solid #ff8500;
  border-radius: 8px;
  overflow: hidden;
  background-color: #2b2b2c;
`;

export const CollapsibleHeader = styled.div`
  padding: 16px;
  background-color: #2b2b2c;
  cursor: pointer;
  color: #ff8500;
  font-weight: bold;
  
  &:hover {
    background-color: #3b3b3c;
  }
`;

export const CollapsibleContent = styled.div`
  padding: 16px;
  color: white;
  background-color: #2b2b2c;
`;

export const DownloadButton = styled(Button)`
  background-color: #ff8500;
  color: white;
  font-family: 'Roboto', sans-serif;
  font-size: 0.9rem;
  font-weight: 500;
  padding: 0.6rem 1.2rem;
  border-radius: 6px;
  border: none;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  text-transform: none;
  margin-top: 1rem;
  
  &:hover {
    background-color: #ffa533;
    transform: translateY(-1px);
  }

  &:active {
    transform: translateY(0);
  }
`;

export const StyledForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
  margin: 2rem 0;
`;

export const FormField = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

export const Label = styled.label`
  color: #ff8500;
  font-family: 'Roboto', sans-serif;
  font-size: 0.9rem;
  font-weight: 500;
  margin-bottom: 0.3rem;
`;

export const Select = styled.select`
  background-color: #2b2b2c;
  border: 2px solid #ff8500;
  border-radius: 6px;
  color: white;
  font-family: 'Roboto', sans-serif;
  font-size: 1rem;
  padding: 0.8rem;
  cursor: pointer;
  transition: all 0.2s ease-in-out;

  &:focus {
    outline: none;
    border-color: #ffa533;
    box-shadow: 0 0 0 2px rgba(255, 133, 0, 0.2);
  }

  &:hover {
    border-color: #ffa533;
  }

  option {
    background-color: #2b2b2c;
    color: white;
    padding: 0.5rem;
  }
`;

export const MultiSelect = styled(Select)`
  height: auto;
  min-height: 120px;
  
  option:checked {
    background-color: #ff8500;
    color: white;
  }
`;

export const SubmitButton = styled(Button)`
  background-color: #ff8500;
  color: white;
  font-family: 'Roboto', sans-serif;
  font-size: 1rem;
  font-weight: 500;
  padding: 0.8rem 1.5rem;
  border-radius: 6px;
  border: none;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  text-transform: none;
  align-self: flex-start;
  margin-top: 1rem;

  &:hover {
    background-color: #ffa533;
    transform: translateY(-1px);
  }

  &:active {
    transform: translateY(0);
  }

  &:disabled {
    background-color: #666;
    cursor: not-allowed;
  }
`;

export const StyledCircularProgress = styled(CircularProgress)`
  color: #ff8500;
`;

export const ErrorMessage = styled.div`
  color: #ff4444;
  background-color: rgba(255, 68, 68, 0.1);
  border: 1px solid #ff4444;
  border-radius: 6px;
  padding: 1rem;
  margin: 1rem 0;
  font-family: 'Roboto', sans-serif;
  font-size: 0.9rem;
`;

export const ValidationMessage = styled.span`
  color: #ff4444;
  font-size: 0.8rem;
  margin-top: 0.3rem;
  font-family: 'Roboto', sans-serif;
`;

export const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 2rem;
  background-color: rgba(43, 43, 44, 0.8);
  border-radius: 8px;
  backdrop-filter: blur(4px);
`;
