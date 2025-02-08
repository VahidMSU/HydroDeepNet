import { styled } from '@mui/material/styles';
import { Box, Container, Typography } from '@mui/material';
import theme from './variables.ts';

import './variables.ts'; // Ensure this path is correct or the file exists

const Body = styled('body')({
  backgroundColor: '#f0f2f5',
  paddingTop: '20px',
  textSizeAdjust: '100%',
});

const SignUpContainer = styled(Container)({
  width: '100%',
  maxWidth: '400px',
  margin: '0 auto',
  backgroundColor: '#ffffff',
  padding: '20px',
  borderRadius: '10px',
  boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
  backdropFilter: 'blur(10px)',
});

const ErrorMessage = styled(Typography)({
  color: 'red',
  fontSize: '0.9rem',
});

const Footer = styled(Box)({
  textAlign: 'center',
  marginTop: theme.spacing(2.5),
  fontSize: '0.9rem',
  color: '#555',
  '& a': {
    color: theme.palette.primary.main,
    textDecoration: 'none',
    '&:hover': {
      textDecoration: 'underline',
    },
  },
});
