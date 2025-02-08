import { styled } from '@mui/material/styles';
import { Box, Button, Container, TextField } from '@mui/material';
import theme from './variables.ts';

const ViewDiv = styled(Box)({
  width: '110%',
  height: 'calc(100vh - 150px)',
  border: 'none',
  borderRadius: '15px',
  boxShadow: '0 8px 30px rgba(0, 0, 0, 0.12)',
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: '0 12px 40px rgba(0, 0, 0, 0.15)',
  },
});

const StyledContainer = styled(Container)({
  maxWidth: '1400px',
  margin: '30px auto',
  padding: '25px',
  backgroundColor: '#f8f9fa',
  borderRadius: '20px',
  boxShadow: '0 10px 40px rgba(0, 0, 0, 0.08)',
  backdropFilter: 'blur(10px)',
});

const Card = styled(Box)({
  background: 'white',
  border: 'none',
  borderRadius: '12px',
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.06)',
  marginBottom: '25px',
  transition: 'transform 0.2s ease, boxShadow 0.2s ease',
  '&:hover': {
    transform: 'translateY(-3px)',
    boxShadow: '0 6px 25px rgba(0, 0, 0, 0.1)',
  },
});

const CardBody = styled(Box)({
  padding: '20px',
});

const FormGroup = styled(Box)({
  width: '90%',
  margin: '0 auto 20px',
  '& label': {
    fontWeight: 600,
    color: '#2c3e50',
    marginBottom: '8px',
    fontSize: '0.95rem',
    textAlign: 'match-parent',
  },
});

const FormControl = styled(TextField)({
  border: '2px solid #e9ecef',
  borderRadius: '8px',
  padding: '12px',
  fontSize: '1rem',
  transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
  '&:focus': {
    borderColor: '#4a90e2',
    boxShadow: '0 0 0 3px rgba(74, 144, 226, 0.1)',
    outline: 'none',
  },
});

const StyledButton = styled(Button)({
  padding: '12px 24px',
  borderRadius: '8px',
  border: 'none',
  background: 'linear-gradient(145deg, #4a90e2, #357abd)',
  color: 'white',
  fontWeight: 600,
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 4px 15px rgba(74, 144, 226, 0.4)',
  },
});

const LoadingIndicator = styled(Box)({
  '& img': {
    animation: 'spin 1s ease-in-out infinite',
  },
});

const Row = styled(Box)({
  display: 'flex',
  flexWrap: 'wrap',
  gap: '30px',
  margin: '0',
});

const ColLg8 = styled(Box)({
  flex: '1 1 65%',
  minWidth: '320px',
});

const ColLg4 = styled(Box)({
  flex: '1 1 30%',
  minWidth: '300px',
});

const ResponsiveLayout = styled('div')({
  '@media (max-width: 768px)': {
    '#viewDiv': {
      height: 'calc(60vh - 100px)',
      width: '120%',
    },
    '.container': {
      padding: theme.spacing(2),
      margin: theme.spacing(2),
    },
    '.row': {
      gap: theme.spacing(2.5),
      flexDirection: 'column',
    },
    '.form-group': {
      width: '100%',
    },
  },
});

const Body = styled('body')({
  textSizeAdjust: '100%',
});

export {
  ViewDiv,
  StyledContainer,
  Card,
  CardBody,
  FormGroup,
  FormControl,
  StyledButton,
  LoadingIndicator,
  Row,
  ColLg8,
  ColLg4,
  ResponsiveLayout,
  Body,
};
