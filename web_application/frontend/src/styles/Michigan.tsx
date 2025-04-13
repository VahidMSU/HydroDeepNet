/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';
import { TitleBase, SectionContainer, ContentWrapper } from './common.tsx';

export const Container = styled.div`
  padding: 2rem;
  background-color: ${colors.background};
`;

export const HeaderTitle = styled.h1`
  text-align: center;
  margin-bottom: 1.8rem;
  color: ${colors.accent};
  font-size: 2.5rem;
  font-weight: bold;
  position: relative;
  padding-bottom: 1.2rem;
  
  &:after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, ${colors.accent}, ${colors.accentHover});
    border-radius: 3px;
  }
`;

// Use common components
export const MichiganContainer = SectionContainer;
export const MichiganTitle = TitleBase;

// Re-export common wrapper
export { ContentWrapper };

export const ImageGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  width: 100%;
`;

export const ImageCard = styled.div`
  border-radius: 12px;
  overflow: hidden;
  background: linear-gradient(145deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
  border: 1px solid ${colors.border};
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
    border-color: ${colors.borderLight};
  }
`;

export const Modal = styled.div`
  display: none; /* Hidden by default */
  position: fixed;
  z-index: 1000;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: ${colors.overlayBg};
  align-items: center;
  justify-content: center;
`;

export const ModalClose = styled.span`
  position: absolute;
  top: 20px;
  right: 20px;
  color: ${colors.text};
  font-size: 2rem;
  cursor: pointer;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background-color: rgba(0, 0, 0, 0.3);
  transition: all 0.2s ease;
  
  &:hover {
    background-color: ${colors.accent};
    color: ${colors.textInverse};
  }
`;

export const CardTitle = styled.h4`
  padding: 1rem;
  color: ${colors.accent};
  font-size: 1.1rem;
  margin: 0;
  text-align: center;
`;

export const ImageElement = styled.img`
  width: 100%;
  height: auto;
  transition: transform 0.3s ease;
  
  ${ImageCard}:hover & {
    transform: scale(1.05);
  }
`;

export const ModalImage = styled.img`
  max-width: 90vw;
  max-height: 90vh;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
`;
