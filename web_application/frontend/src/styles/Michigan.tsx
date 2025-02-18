import styled from 'styled-components';
import colors from './colors.tsx';

export const Container = styled.div`
  padding: 20px;
`;

export const HeaderTitle = styled.h1`
  text-align: center;
  margin-bottom: 30px;
  color: ${colors.TitleText};
  font-size: 2.5rem;
  font-weight: bold;
`;

export const MichiganContainer = styled.div`
  max-width: 1200px;
  margin: 2rem auto;
  padding: 2rem;
  color: #ffffff;
`;

export const MichiganTitle = styled.h2`
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
