/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';
import { ContentWrapper, SectionHeader, TitleBase, SectionContainer, StyledLinks } from './common.tsx';

// Remove legacy style objects if not used elsewhere
// ...existing code if needed...

// Use common components to reduce duplication
export const PrivacyContainer = SectionContainer;
export const PrivacyTitle = TitleBase;

export const PolicyText = styled.p`
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
