/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import { Box, Button } from '@mui/material';
//import './variables.ts'; // Ensure this path is correct or the file exists

// Main layout container
export const PageLayout = styled.div`
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 20px;
  height: 100vh;
  padding: 20px;
  background-color: #2b2b2c;
`;

// Left sidebar container
export const Sidebar = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
  overflow-y: auto;
  background-color: #444e5e;
  padding: 20px;
  border-radius: 12px;
`;

// Map container
export const MapContainer = styled.div`
  height: 100%;
  background: #444e5e;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  overflow: hidden;
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

export const Result = styled(Box)({
  marginTop: 'var(--spacing)',
  padding: 'var(--spacing)',
  backgroundColor: 'var(--secondary-color)',
  borderRadius: 'var(--radius)',
});

export const ViewDiv = styled(Box)({
  width: '120%',
  height: 'calc(100vh - 100px)',
  border: 'none',
  borderRadius: '15px',
  boxShadow: '0 8px 30px rgba(0, 0, 0, 0.12)',
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: '0 12px 40px rgba(0, 0, 0, 0.15)',
  },
});

export const Container = styled.div`
  margin: 40px auto;
  max-width: 1200px;
  padding: 20px;
  background-color: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
`;

export const DataDisplay = styled.div`
  padding: 20px;
  background-color: #fff;
  border-radius: 12px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);

  pre {
    margin: 0;
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 8px;
    overflow-x: auto;
  }
`;

export const Title = styled.h3`
  margin-bottom: 20px;
  color: #ff8500;
  font-size: 1.2rem;
`;

export const SectionHeader = styled.h2`
  color: #ff8500;
  font-size: 1.5rem;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #ff8500;
`;

export const Collapsible = styled.div`
  margin-bottom: 1rem;
  border: 1px solid #ff8500;
  border-radius: 8px;
  overflow: hidden;
  background-color: #2b2b2c;
`;

export const CollapsibleHeader = styled.div`
  padding: 1rem;
  background-color: #2b2b2c;
  cursor: pointer;
  display: flex;
  align-items: center;
  font-weight: 500;
  color: #ff8500;
  
  &:hover {
    background-color: #3b3b3c;
  }
`;

export const CollapsibleContent = styled.div`
  padding: 1rem;
  background-color: #2b2b2c;
  color: white;
`;

export const InfoSection = styled.div`
  margin: 1rem 0;
  padding: 1rem;
  background-color: #2b2b2c;
  border-radius: 8px;
  border: 1px solid #ff8500;
`;

export const InfoContent = styled.div`
  color: white;
  
  p {
    margin-bottom: 1rem;
  }
  
  ul {
    margin-left: 1.5rem;
    margin-bottom: 1rem;
  }
  
  li {
    margin-bottom: 0.5rem;
  }

  strong {
    color: #ff8500;
  }
`;

export const DataResults = styled.div`
  background-color: #2b2b2c;
  border-radius: 8px;
  padding: 1.5rem;
  margin-top: 1.5rem;
  color: white;
  border: 2px solid #ff8500;
  font-family: 'Roboto Mono', monospace;
  
  pre {
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    background-color: #1b1b1c;
    padding: 1.2rem;
    border-radius: 6px;
    font-size: 0.9rem;
    line-height: 1.5;
  }
`;

export const StyledForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
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

export const Input = styled.input`
  background-color: #2b2b2c;
  border: 2px solid #ff8500;
  border-radius: 6px;
  color: white;
  font-family: 'Roboto', sans-serif;
  font-size: 1rem;
  padding: 0.8rem;
  transition: all 0.2s ease-in-out;

  &:focus {
    outline: none;
    border-color: #ffa533;
    box-shadow: 0 0 0 2px rgba(255, 133, 0, 0.2);
  }

  &:hover {
    border-color: #ffa533;
  }

  &::placeholder {
    color: #666;
  }
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
    transform: none;
  }
`;

export const ReadOnlyInput = styled(Input)`
  background-color: #3b3b3c;
  cursor: not-allowed;
  opacity: 0.8;
  border-color: #666;

  &:hover {
    border-color: #666;
  }
`;

export const CoordinatesDisplay = styled.div`
  background-color: #3b3b3c;
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem 0;
  border: 1px solid #666;
`;

export const CoordinateField = styled.div`
  margin-bottom: 0.8rem;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

export const CoordinateLabel = styled.div`
  color: #ff8500;
  font-size: 0.9rem;
  font-weight: 500;
  margin-bottom: 0.4rem;
`;