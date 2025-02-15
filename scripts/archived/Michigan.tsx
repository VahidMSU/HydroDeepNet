/** @jsxImportSource @emotion/react */
import { styled } from '@mui/material/styles';
import { Box } from '@mui/material';
import theme from './variables.ts';
import './variables.ts'; // Ensure this path is correct or the file exists

const Container = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  margin: theme.spacing(2.5, 'auto'),
  maxWidth: '1200px',
  padding: theme.spacing(2.5),
});

export { Container };
