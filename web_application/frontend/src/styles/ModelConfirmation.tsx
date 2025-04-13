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
  padding: 2rem;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

export const Card = styled.div`
  background: linear-gradient(145deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 12px;
  overflow: hidden;
  width: 90%;
  max-width: 800px;
  margin: 2rem auto;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  border: 1px solid ${colors.border};
  position: relative;
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(to right, ${colors.success}, transparent);
  }
  
  .alert {
    padding: 2rem;
    
    &.alert-success {
      background: transparent;
      border: none;
      color: ${colors.text};
    }
    
    .alert-heading {
      color: ${colors.success};
      font-size: 1.8rem;
      margin-bottom: 1.2rem;
      display: flex;
      align-items: center;
      gap: 0.8rem;
    }
    
    hr {
      border: none;
      border-top: 1px solid ${colors.border};
      margin: 1.5rem 0;
    }
    
    p {
      color: ${colors.textSecondary};
      line-height: 1.6;
    }
  }
`;

export const BtnPrimary = styled.button`
  background: linear-gradient(to right, ${colors.accent}, ${colors.accentHover});
  color: ${colors.textInverse};
  border: none;
  border-radius: 8px;
  padding: 0.8rem 1.5rem;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: inline-flex;
  align-items: center;
  gap: 0.8rem;
  box-shadow: 0 4px 12px rgba(255, 133, 0, 0.3);
  
  &:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(255, 133, 0, 0.4);
  }
  
  i {
    font-size: 1.2rem;
  }
`;

export default theme;
