import { styled } from '@mui/material/styles';
import { Box, Button } from '@mui/material';
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#d2691e',
      dark: '#a0522d',
    },
    secondary: {
      main: '#2f4f4f',
      dark: '#1e3d3d',
    },
    background: {
      default: '#1e1e1e',
      paper: '#2d2d2d',
    },
    text: {
      primary: '#dcdcdc',
      secondary: '#a9a9a9',
    },
    error: {
      main: '#ff6347',
    },
  },
  spacing: 4,
  shape: {
    borderRadius: 8,
  },
  shadows: [
    'none',
    '0 2px 4px rgba(0, 0, 0, 0.05)',
    '0 4px 6px rgba(0, 0, 0, 0.1)',
    '0 10px 15px rgba(0, 0, 0, 0.1)',
    '0 20px 25px rgba(0, 0, 0, 0.1)',
    '0 25px 30px rgba(0, 0, 0, 0.1)',
    '0 30px 35px rgba(0, 0, 0, 0.1)',
    '0 35px 40px rgba(0, 0, 0, 0.1)',
    '0 40px 45px rgba(0, 0, 0, 0.1)',
    '0 45px 50px rgba(0, 0, 0, 0.1)',
    '0 50px 55px rgba(0, 0, 0, 0.1)',
    '0 55px 60px rgba(0, 0, 0, 0.1)',
    '0 60px 65px rgba(0, 0, 0, 0.1)',
    '0 65px 70px rgba(0, 0, 0, 0.1)',
    '0 70px 75px rgba(0, 0, 0, 0.1)',
    '0 75px 80px rgba(0, 0, 0, 0.1)',
    '0 80px 85px rgba(0, 0, 0, 0.1)',
    '0 85px 90px rgba(0, 0, 0, 0.1)',
    '0 90px 95px rgba(0, 0, 0, 0.1)',
    '0 95px 100px rgba(0, 0, 0, 0.1)',
    '0 100px 105px rgba(0, 0, 0, 0.1)',
    '0 105px 110px rgba(0, 0, 0, 0.1)',
    '0 110px 115px rgba(0, 0, 0, 0.1)',
    '0 115px 120px rgba(0, 0, 0, 0.1)',
    '0 120px 125px rgba(0, 0, 0, 0.1)',
  ],
  typography: {
    fontFamily: 'system-ui, -apple-system, sans-serif',
    fontSize: 14,
    fontWeightRegular: 400,
    fontWeightMedium: 500,
    fontWeightBold: 600,
  },
});

export default theme;


const Body = styled('body')({
  backgroundColor: '#f8f9fa',
  textSizeAdjust: '100%',
});

const Card = styled(Box)({
  border: 'none',
  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
  backdropFilter: 'blur(10px)',
});

const BtnPrimary = styled(Button)({
  display: 'flex',
  alignItems: 'center',
  textAlign: 'match-parent',
  '& i': {
    marginRight: theme.spacing(0.5),
  },
});
