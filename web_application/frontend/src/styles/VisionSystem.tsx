/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';  

const VisionSystemContainer = styled.div`
  /* Add your styles here */
`;

const ContainerFluid = styled.div`
  padding: 2rem;
  background-color: ${colors.background};
  max-width: 1400px;
  margin: 0 auto;
`;

const CardBody = styled.div`
  padding: 2rem;
  background-color: ${colors.surface};
  color: ${colors.text};
  @media (max-width: 768px) {
    padding: 1rem;
  }
`;

export const VisionContainer = styled.div`
  max-width: 1200px;
  margin: 2rem auto;
  padding: 2rem;
  color: #ffffff;
`;

export const VisionTitle = styled.h2`
  color: #ff8500;
  font-size: 2.8rem;
  margin-bottom: 1.5rem;
  border-bottom: 3px solid #ff8500;
  padding-bottom: 1.2rem;
  position: relative;
  text-align: center;
  
  &:after {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background-color: #ffa533;
  }
`;

export const ContentWrapper = styled.div`
  background-color: #444e5e;
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
`;

export const SectionHeader = styled.h3`
  color: #ff8500;
  font-size: 1.8rem;
  margin: 2rem 0 1rem;
`;

export const SectionText = styled.p`
  color: #cccccc;
  font-size: 1.2rem;
  line-height: 1.8;
  margin-bottom: 1.5rem;
  opacity: 0.9;
`;

const VideoGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: repeat(2, 1fr);
  gap: 2rem;
  margin: 2rem 0;
  
  @media (max-width: 1200px) {
    grid-template-columns: repeat(2, 1fr);
    grid-template-rows: repeat(3, 1fr);
  }
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    grid-template-rows: repeat(6, 1fr);
  }
`;

export const VideoContainer = styled.div`
  height: 300px;
  width: 100%;
  border-radius: 12px;
  overflow: hidden;
  transition: transform 0.3s ease;
  background-color: rgba(255, 255, 255, 0.1);
  cursor: pointer;

  &:hover {
    transform: translateY(-5px);
  }

  video, img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    background-color: rgba(0, 0, 0, 0.2);
  }

  &.main-video {
    height: 400px;
    margin-bottom: 2rem;
  }
`;

const TextCenter = styled.div`
  text-align: center;
  margin: 2rem 0;
`;

export const MediaModal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
`;

export const MediaWrapper = styled.div`
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
  display: flex;
  justify-content: center;
  align-items: center;

  video, img {
    max-width: 100%;
    max-height: 90vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
  }

  video {
    background-color: rgba(0, 0, 0, 0.5);
  }
`;

const NavigationButton = styled.button`
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  background: ${colors.overlayBg};
  border: none;
  color: ${colors.text};
  padding: 1rem;
  cursor: pointer;
  border-radius: 50%;
  transition: background 0.3s ease;

  &:hover {
    background: ${colors.surfaceLight};
  }

  &.prev {
    left: 20px;
  }

  &.next {
    right: 20px;
  }
`;

const CloseButton = styled.button`
  position: absolute;
  top: 20px;
  right: 20px;
  background: transparent;
  border: none;
  color: ${colors.text};
  font-size: 2rem;
  cursor: pointer;
  z-index: 1001;
`;

export { 
  VisionSystemContainer, 
  ContainerFluid, 
  CardBody, 
  VideoGrid, 
  TextCenter, 
  NavigationButton, 
  CloseButton 
};
