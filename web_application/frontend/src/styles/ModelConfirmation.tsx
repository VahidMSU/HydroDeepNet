import { styled } from '@mui/material/styles';
import { Box, Button } from '@mui/material';
import theme from './variables';

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
