/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';
import { TitleBase, SectionContainer, ContentWrapper } from './common.tsx';

export const Container = styled.div`
  padding: 20px;
`;

export const HeaderTitle = styled.h1`
  text-align: center;
  margin-bottom: 30px;
  color: ${colors.text};
  font-size: 2.5rem;
  font-weight: bold;
`;

// Use common components
export const MichiganContainer = SectionContainer;
export const MichiganTitle = TitleBase;

// Re-export common wrapper
export { ContentWrapper };

export const ImageGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  width: 100%;
`;

export const ImageCard = styled.div`
  border-radius: 12px;
  overflow: hidden;
  background-color: rgba(255, 255, 255, 0.1);
  transition: transform 0.3s ease;

  &:hover {
    transform: translateY(-5px);
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
`;

export const CardTitle = styled.h4`
  padding: 1rem;
  color: #ff8500;
  font-size: 1.1rem;
  margin: 0;
  text-align: center;
`;

export const ImageElement = styled.img`
  width: 100%;
  height: auto;
`;

export const ModalImage = styled.img`
  max-width: 90vw;
  max-height: 90vh;
  object-fit: contain;
`;
