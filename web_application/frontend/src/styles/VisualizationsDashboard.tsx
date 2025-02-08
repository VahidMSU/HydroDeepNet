import { styled } from '@mui/material/styles';
import { Box, Button, Container, TextField, Typography } from '@mui/material';
import theme from './variables.ts';

const Body = styled('body')({
  backgroundColor: '#f0f2f5',
  color: '#1a1a1a',
  lineHeight: 1.6,
  textSizeAdjust: '100%',
});

const FormContainer = styled(Container)({
  backgroundColor: '#ffffff',
  borderRadius: '16px',
  padding: '2.5rem',
  boxShadow: '0 4px 24px rgba(0, 0, 0, 0.06)',
  margin: '2rem auto',
  maxWidth: '1200px',
});

const FormLabel = styled('label')({
  display: 'block',
  marginBottom: '8px',
  fontWeight: 600,
  fontSize: '0.9rem',
  color: '#4a5568',
});

const FormControl = styled(TextField)({
  width: '100%',
  padding: '0.75rem 1rem',
  border: '2px solid #e2e8f0',
  borderRadius: '8px',
  fontSize: '1rem',
  transition: 'all 0.2s ease',
  backgroundColor: '#f8fafc',
  marginBottom: '1rem',
  '&:focus': {
    borderColor: '#3b82f6',
    boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)',
    outline: 'none',
  },
});

const FormGrid = styled(Box)({
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
  gap: '2rem',
  marginTop: '2rem',
  backdropFilter: 'blur(10px)',
});

const FormGroup = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  gap: '0.5rem',
});

const Btn = styled(Button)({
  padding: '0.75rem 1.5rem',
  borderRadius: '8px',
  fontWeight: 600,
  transition: 'all 0.2s ease',
  border: 'none',
  cursor: 'pointer',
});

const BtnPrimary = styled(Btn)({
  backgroundColor: '#3b82f6',
  color: 'white',
  '&:hover': {
    backgroundColor: '#2563eb',
    transform: 'translateY(-1px)',
  },
});

const FormError = styled(Typography)({
  color: '#ef4444',
  fontSize: '0.875rem',
  marginTop: '0.25rem',
});

const VisualizationResults = styled(Box)({
  marginTop: '2rem',
});

const GifContainer = styled(Box)({
  display: 'flex',
  flexWrap: 'wrap',
  justifyContent: 'center',
  gap: '1rem',
});

const GifWrapper = styled(Box)({
  maxWidth: '300px',
  border: '1px solid #ddd',
  borderRadius: '8px',
  overflow: 'hidden',
  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
  '& img': {
    width: '100%',
    height: 'auto',
  },
});

const ResponsiveDesign = styled('div')({
  '@media (max-width: 768px)': {
    '.form-container': {
      padding: theme.spacing(3),
      margin: theme.spacing(2),
    },
    '.form-actions': {
      flexDirection: 'column',
    },
    '.btn': {
      width: '100%',
      marginBottom: theme.spacing(1),
    },
  },
});

export {
  Body,
  FormContainer,
  FormLabel,
  FormControl,
  FormGrid,
  FormGroup,
  BtnPrimary,
  FormError,
  VisualizationResults,
  GifContainer,
  GifWrapper,
  ResponsiveDesign,
};
