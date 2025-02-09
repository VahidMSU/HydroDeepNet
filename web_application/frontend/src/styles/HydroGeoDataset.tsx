/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import { Box } from '@mui/material';
import './variables.ts'; // Ensure this path is correct or the file exists

// Main layout container
export const PageLayout = styled.div`
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 20px;
  height: 100vh;
  padding: 20px;
  background-color: #f5f5f5;
`;

// Left sidebar container
export const Sidebar = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
  overflow-y: auto;
`;

// Map container
export const MapContainer = styled.div`
  height: 100%;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  overflow: hidden;
`;

export const FormContainer = styled(Box)({
  padding: '20px',
  border: '1px solid var(--border-color)',
  borderRadius: '12px',
  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
  backgroundColor: '#fff',
  transition: 'box-shadow 0.3s ease',
  '&:hover': {
    boxShadow: '0 6px 16px rgba(0, 0, 0, 0.12)',
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
  color: #333;
  font-size: 1.2rem;
`;