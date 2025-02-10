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
  marginBottom: '1rem',
  fontWeight: 600,
  fontSize: '1rem',
  color: '#4a5568',
  letterSpacing: '0.01em',
  textAlign: 'center',  // Center the label text
  width: '100%',        // Full width for center alignment
  paddingLeft: '0',     // Remove left padding
  position: 'relative',
  '&::before': {
    display: 'none'     // Remove the marker since we're centering
  }
});

const FormControl = styled(TextField)({
  width: '100%',
  padding: '0.75rem 1rem',
  border: '2px solid #e2e8f0',
  borderRadius: '8px',
  fontSize: '1rem',
  transition: 'all 0.2s ease',
  backgroundColor: '#f8fafc',
  marginBottom: '1.5rem',
  marginLeft: '0',      // Remove left margin
  maxWidth: '300px',    // Limit maximum width
  textAlign: 'center',  // Center the text inside
  '&:focus': {
    borderColor: '#3b82f6',
    boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)',
    outline: 'none',
  },
  '& .MuiInputBase-root': {
    marginTop: '0.5rem',  // Added space between label and input
  },
});

const FormGrid = styled(Box)({
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
  gap: '4rem',  // Increased gap between grid items
  marginTop: '3rem',    // Increased top margin
  marginBottom: '3rem', // Increased bottom margin
  backdropFilter: 'blur(10px)',
  maxWidth: '900px',    // Increased max width to accommodate spacing
  margin: '0 auto',
  justifyContent: 'center',
  alignItems: 'start',
  padding: '2rem',      // Added padding around the grid
});

const FormGroup = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  gap: '1.5rem',        // Increased gap between label and input
  marginBottom: '2rem', // Increased bottom margin
  padding: '1.5rem',    // Added padding around each form group
  alignItems: 'center',
  textAlign: 'center',
  backgroundColor: 'rgba(255, 255, 255, 0.5)', // Optional: subtle background
  borderRadius: '12px', // Optional: rounded corners
  '& .MuiFormControl-root': {
    marginBottom: '2rem',
  },
  '&.ensemble-group': {
    '& .MuiFormControl-root': {
      marginLeft: '0',  // Remove left margin since we're centering
      width: '100%',    // Use full width
      maxWidth: '300px' // Limit maximum width
    },
    '& label': {
      marginLeft: '0',  // Remove left margin
    }
  }
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
  marginTop: '1.5rem',  // Added top margin for button
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
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
  gap: '2rem',
  padding: '1rem',
  width: '100%',
});

const GifWrapper = styled(Box)({
  position: 'relative',
  borderRadius: '12px',
  overflow: 'hidden',
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
  backgroundColor: '#ffffff',
  '& img': {
    width: '100%',
    height: 'auto',
    display: 'block',
  },
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 8px 12px rgba(0, 0, 0, 0.15)',
  },
});

const PageTitle = styled(Typography)({
  fontSize: '2.5rem',
  fontWeight: 700,
  textAlign: 'center',
  marginBottom: '2rem',
  color: '#2d3748',
  width: '100%',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  padding: '1rem 0',
  '@media (max-width: 768px)': {
    fontSize: '2rem',
  },
});

const SectionTitle = styled(Typography)({
  fontSize: '1.8rem',
  fontWeight: 600,
  textAlign: 'center',
  marginBottom: '1.5rem',
  color: '#2d3748',
});

const FormSelect = styled(TextField)({
  width: '100%',
  padding: '0.75rem 1rem',
  border: '2px solid #e2e8f0',
  borderRadius: '8px',
  fontSize: '1rem',
  transition: 'all 0.2s ease',
  backgroundColor: '#f8fafc',
  marginBottom: '1.5rem',
  marginLeft: '0',      // Remove left margin
  maxWidth: '300px',    // Limit maximum width
  textAlign: 'center',  // Center the text inside
  '&:focus': {
    borderColor: '#3b82f6',
    boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)',
    outline: 'none',
  },
  '& .MuiInputBase-root': {
    marginTop: '0.5rem',  // Added space between label and input
  },
  '& .MuiSelect-select': {
    padding: '0.75rem 1rem',
  },
  '& .MuiFormHelperText-root': {
    marginTop: '0.5rem',  // Added space for helper text
  },
});

const NoResults = styled(Typography)({
  textAlign: 'center',
  color: '#64748b',
  fontSize: '1.1rem',
  padding: '2rem',
  width: '100%',
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
  PageTitle,
  SectionTitle,
  FormSelect,
  NoResults,
  ResponsiveDesign,
};
