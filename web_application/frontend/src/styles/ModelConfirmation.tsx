/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import { Button } from '@mui/material';
import colors from './colors.tsx';
import { createTheme } from '@mui/material/styles';

// Keep theme definition for MUI components that require it
const theme = createTheme({
  palette: {
    primary: {
      main: colors.accent,
      dark: colors.accentHover,
    },
    secondary: {
      main: colors.accentAlt,
      dark: colors.surfaceDark,
    },
    background: {
      default: colors.background,
      paper: colors.surface,
    },
    text: {
      primary: colors.text,
      secondary: colors.textSecondary,
    },
    error: {
      main: colors.error,
    },
  },
  // Only keep essential theme settings
  spacing: 4,
  shape: {
    borderRadius: 8,
  },
  typography: {
    fontFamily: 'system-ui, -apple-system, sans-serif',
    fontSize: 14,
    fontWeightRegular: 400,
    fontWeightMedium: 500,
    fontWeightBold: 600,
  },
});

// Simplified styled components 
export const Body = styled.div`
  background-color: ${colors.background};
  text-size-adjust: 100%;
`;

export const Card = styled.div`
  border: 1px solid ${colors.border};
  box-shadow: 0 4px 8px ${colors.overlayBg};
  backdrop-filter: blur(10px);
  background-color: ${colors.surface};
  border-radius: 8px;
  padding: 16px;
`;

export const BtnPrimary = styled(Button)`
  display: flex;
  align-items: center;
  text-align: match-parent;
  background-color: ${colors.accent};
  color: ${colors.text};
  &:hover {
    background-color: ${colors.accentHover};
  }
  & i {
    margin-right: 4px;
  }
`;

export default theme;
