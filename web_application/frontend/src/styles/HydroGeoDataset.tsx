/** @jsxImportSource @emotion/react */
import { styled } from '@mui/material/styles';
import { Box } from '@mui/material';
import theme from './variables.ts';
import './variables.ts'; // Ensure this path is correct or the file exists

export const FormContainer = styled(Box)({
  flex: 1,
  maxWidth: '400px',
  minWidth: '300px',
  padding: 'var(--spacing)',
  border: '1px solid var(--border-color)',
  borderRadius: 'var(--radius)',
  boxShadow: 'var(--shadow)',
  backgroundColor: '#fff',
  transition: 'box-shadow 0.3s ease',
  backdropFilter: 'blur(10px)',
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
