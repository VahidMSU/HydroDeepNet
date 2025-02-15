/** @jsxImportSource @emotion/react */
import { styled } from '@mui/material/styles';
import { Box } from '@mui/material';
import theme from './variables.ts';
import './variables.ts'; // Ensure this path is correct or the file exists

const Body = styled('body')({
  textSizeAdjust: '100%',
});

const Container = styled(Box)({
  maxWidth: '600px',
  margin: 'auto',
  padding: theme.spacing(2.5),
  textAlign: 'match-parent',
  backdropFilter: 'blur(10px)',
});

export { Body, Container };
