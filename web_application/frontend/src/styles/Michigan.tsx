import styled from 'styled-components';

export const Container = styled.div`
  padding: 20px;
`;

export const HeaderTitle = styled.h1`
  text-align: center;
  margin-bottom: 30px;
  color: #333;
  font-size: 2.5rem;
  font-weight: bold;
`;

export const ImageGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  width: 100%;
`;

export const ImageCard = styled.div`
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  background-color: #fff;
  overflow: hidden;
  text-align: center;
  cursor: pointer;

  img {
    width: 100%;
    height: auto;
  }

  h4 {
    padding: 10px;
    background-color: #f9f9f9;
    font-size: 1.1rem;
    color: #555;
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
  background-color: rgba(0, 0, 0, 0.8);
  align-items: center;
  justify-content: center;
`;

export const ModalClose = styled.span`
  position: absolute;
  top: 20px;
  right: 20px;
  color: white;
  font-size: 2rem;
  cursor: pointer;
`;

export const CardTitle = styled.h4`
  padding: 10px;
  background-color: #f9f9f9;
  font-size: 1.1rem;
  color: #555;
  margin: 0;
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
