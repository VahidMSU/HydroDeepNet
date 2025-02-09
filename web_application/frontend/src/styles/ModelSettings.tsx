/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';

const ContainerFluid = styled.div`
  padding: 15px 25px 15px 15px;  // Added right padding
  background-color: #f8f9fa;
  min-height: 100vh;
  max-width: 100vw;
`;

export const ContentWrapper = styled.div`
  flex-grow: 1;
  margin-left: 270px;
  padding: 2.5rem;
  background-color: #ffffff;
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
  overflow-y: auto;
  height: calc(100vh - 220px);
  transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);

  &:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
  }

  @media (max-width: 768px) {
    margin-left: 0;
    height: auto;
    max-height: 700px;
  }
`;

export const FooterContainer = styled.footer`
  width: 10%;
  margin-left: 0;
  background-color: #343a40;
  color: #343a40;
  text-align: center;
  padding: 15px;
  font-size: 10px;
  border-top: 1px solid #343a40;
  position: fixed;
  bottom: 0;

  a {
    color: #d8dbde;
    text-decoration: none;
    margin: 0 10px;
    font-weight: 500;

    &:hover {
      text-decoration: underline;
    }
  }
`;

const Title = styled.h2`
  text-align: center;
  margin: 40px 0;
  color: #333;
`;

const Row = styled.div`
  display: flex;
  flex-wrap: wrap;
  width: 100%;
  height: 100%;
  gap: 1.5rem;  // Increased gap between columns
  margin: 0;
  padding: 0;  // Removed small horizontal padding
`;

export const Content = styled.main`
  margin-left: 65px;  // Reduced margin to move closer to sidebar
  padding: 0 20px 0 0;  // Added right padding
  display: flex;
  flex-wrap: wrap;
  width: calc(100vw - 85px);  // Adjusted for new spacing
  height: calc(100vh - 30px); // Minimal spacing for header
  overflow: hidden;
  box-sizing: border-box;

  @media (max-width: 768px) {
    margin-left: 0;
    width: 100vw;
    padding: 0 10px;
  }
`;

interface ColumnProps {
  width?: number;
  minWidth?: string;
  mobileMinWidth?: string;
}

const Column = styled.div<ColumnProps>`
  flex: ${props => props.width || 1};
  min-width: ${props => props.minWidth || '400px'};
  height: 100%;
  display: flex;
  flex-direction: column;

  &.map-column {
    position: relative;
    overflow: hidden;
  }

  @media (max-width: 1400px) {
    min-width: ${props => props.mobileMinWidth || '100%'};
  }
`;

export const MapWrapper = styled(ContentWrapper)`
  height: 100% !important;
  width: 100%;
  margin: 0;
  padding: 1rem 1.5rem 1rem 1rem;  // Increased right padding
  position: sticky;
  top: 0;
  overflow: hidden;
  transform: none !important;

  &:hover {
    transform: none !important;
  }

  @media (max-width: 1400px) {
    height: 80vh !important;
  }
`;

const Card = styled.div`
  background-color: #ffffff;
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
  overflow: hidden;
  margin-bottom: 2rem;
  border: 1px solid #eef0f2;
  transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);

  &:hover {
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
    border-color: #e0e4e8;
  }
`;

const CardBody = styled.div`
  padding: 0.75rem;  // Adjusted padding
`;

export { ContainerFluid, Title, Row, Column, Card, CardBody };
