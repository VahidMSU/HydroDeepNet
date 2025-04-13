/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';

export const HomeContainer = styled.div`
  max-width: 1200px;
  margin: 2rem auto;
  padding: 2rem;
  color: ${colors.text};
  background-color: ${colors.background};
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
`;

export const HomeTitle = styled.h2`
  color: ${colors.accent};
  font-size: 2.8rem;
  margin-bottom: 1.8rem;
  border-bottom: 3px solid ${colors.accent};
  padding-bottom: 1.2rem;
  position: relative;
  text-align: center;
  
  &:after {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, ${colors.accent}, ${colors.accentHover});
    border-radius: 3px;
  }
`;

export const ContentWrapper = styled.div`
  background: linear-gradient(145deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 16px;
  padding: 2.2rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  border: 1px solid ${colors.border};
  overflow: hidden;
  position: relative;
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(to right, ${colors.accent}80, transparent);
  }
`;

export const SectionHeader = styled.h3`
  color: ${colors.accent};
  font-size: 1.8rem;
  margin: 2.2rem 0 1.2rem;
  position: relative;
  padding-left: 1rem;
  
  &:before {
    content: '';
    position: absolute;
    left: 0;
    top: 10%;
    height: 80%;
    width: 4px;
    background: linear-gradient(to bottom, ${colors.accent}, transparent);
    border-radius: 2px;
  }
`;

export const SectionText = styled.p`
  color: ${colors.textSecondary};
  font-size: 1.2rem;
  line-height: 1.8;
  margin-bottom: 1.5rem;
  opacity: 0.9;
  
  ul {
    margin: 1rem 0;
    padding-left: 2rem;
  }
  
  li {
    margin-bottom: 0.8rem;
    color: ${colors.textSecondary};
  }
  
  strong {
    color: ${colors.text};
    font-weight: 600;
  }
`;

export const ImageContainer = styled.div`
  height: 300px;
  width: 100%;
  border-radius: 14px;
  overflow: hidden;
  transition: all 0.3s ease;
  background-color: rgba(30, 30, 34, 0.6);
  cursor: pointer;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
  border: 1px solid ${colors.border};
  margin: 1.5rem 0;

  &:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
    border-color: ${colors.borderLight};
  }

  img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    background-color: rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
  }

  &.main-image {
    height: 400px;
    margin-bottom: 2.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
  }
`;

export const CardGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 2rem;
  margin: 2rem 0;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

export const Card = styled.div`
  background-color: ${colors.surfaceLight};
  border-radius: 14px;
  overflow: hidden;
  transition: all 0.3s ease;
  cursor: pointer;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
  border: 1px solid ${colors.border};
  display: flex;
  flex-direction: column;

  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
    border-color: ${colors.borderLight};
  }

  .card-image {
    height: 180px;
    overflow: hidden;
    
    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform 0.3s ease;
    }
  }

  &:hover .card-image img {
    transform: scale(1.05);
  }
  
  .card-content {
    padding: 1.5rem;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
  }
  
  .card-title {
    color: ${colors.accent};
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
  }
  
  .card-description {
    color: ${colors.textSecondary};
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 1.5rem;
    flex-grow: 1;
  }
`;

export const Button = styled.a`
  display: inline-block;
  background-color: ${colors.accent};
  color: #fff;
  font-weight: 600;
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  text-decoration: none;
  text-align: center;
  transition: all 0.2s ease;
  border: none;
  cursor: pointer;
  min-width: 180px;
  
  &:hover {
    background-color: ${colors.accentHover};
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
`;

export const ButtonContainer = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
  justify-content: center;
  
  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

// Modal styles
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

  img {
    max-width: 100%;
    max-height: 90vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
  }
`;

export const CloseButton = styled.button`
  position: absolute;
  top: 20px;
  right: 20px;
  background: transparent;
  border: none;
  color: ${colors.text};
  font-size: 2rem;
  cursor: pointer;
  z-index: 1001;
  
  &:hover {
    color: ${colors.accent};
  }
`; 