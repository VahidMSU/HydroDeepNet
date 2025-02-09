/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';

const VisionSystemContainer = styled.div`
  /* Add your styles here */
`;

const ContainerFluid = styled.div`
  padding: 2rem;
  background-color: #f8f9fa;
  max-width: 1400px;
  margin: 0 auto;
`;

const CardBody = styled.div`
  padding: 2rem;
  @media (max-width: 768px) {
    padding: 1rem;
  }
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

const VideoContainer = styled.div`
  height: 300px;
  width: 100%;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  transition: transform 0.3s ease;

  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
  }

  video {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  @media (max-width: 768px) {
    height: 300px;
  }
`;

const TextCenter = styled.div`
  text-align: center;
  margin: 2rem 0;
`;

const MediaModal = styled.div`
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

const MediaWrapper = styled.div`
  position: relative;
  max-width: 90vw;
  max-height: 90vh;

  video, img {
    max-width: 100%;
    max-height: 90vh;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
  }
`;

const NavigationButton = styled.button`
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  background: rgba(255, 255, 255, 0.2);
  border: none;
  color: white;
  padding: 1rem;
  cursor: pointer;
  border-radius: 50%;
  transition: background 0.3s ease;

  &:hover {
    background: rgba(255, 255, 255, 0.3);
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
  color: white;
  font-size: 2rem;
  cursor: pointer;
  z-index: 1001;
`;

export { 
  VisionSystemContainer, 
  ContainerFluid, 
  CardBody, 
  VideoGrid, 
  VideoContainer, 
  TextCenter, 
  MediaModal, 
  MediaWrapper, 
  NavigationButton, 
  CloseButton 
};
