/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';
import { ContentWrapper, SectionHeader, TitleBase, SectionContainer, StyledLinks } from './common.tsx';

// Fixed styles with proper syntax (semicolons, not commas)
export const bodyStyle = styled.div`
  font-family: 'Arial, sans-serif';
  margin: 40px;
  padding: 20px;
  background-color: ${colors.background};
  color: ${colors.text};
  line-height: 1.6;
`;

export const headerStyle = styled.header` 
  background: ${colors.surfaceDark};
  color: ${colors.text};
  padding: 15px;
  text-align: center;
  border-radius: 5px;
`;

export const mainStyle = styled.main`
  background: ${colors.surface};
  padding: 20px;
  border-radius: 5px;
  box-shadow: 0px 0px 10px ${colors.overlayBg};
`;

export const Body = styled.div`
  font-family: Arial, sans-serif;
  margin: 40px;
  padding: 20px;
  background-color: ${colors.background};
  color: ${colors.text};
  line-height: 1.6;
`;

export const Header = styled.header`
  background: ${colors.surfaceDark};
  color: ${colors.text};
  padding: 15px;
  text-align: center;
  border-radius: 5px;
`;

export const Main = styled.main`
  background: ${colors.surface};
  padding: 20px;
  border-radius: 5px;
  box-shadow: 0px 0px 10px ${colors.overlayBg};
`;

// Use common base instead of duplicating styling
export const TermsContainer = SectionContainer;
export const TermsTitle = TitleBase;

export const TermsText = styled.p`
  color: #cccccc;
  font-size: 1.2rem;
  line-height: 1.8;
  margin-bottom: 1.5rem;
  opacity: 0.9;
`;

export const ContactSection = styled.div`
  margin-top: 2rem;
  text-align: center;
`;

// Use common links component
export const Links = StyledLinks;

// Re-export common components
export { ContentWrapper, SectionHeader };
