/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';

const ContainerFluid = styled.div`
  padding: 20px;
  background-color: #f8f9fa;
`;

const Title = styled.h2`
  text-align: center;
  margin: 40px 0;
  color: #333;
`;

const Row = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
`;

const Column = styled.div`
  flex: 1;
  min-width: 300px;
  display: flex;
  flex-direction: column;
  align-items: stretch;
`;

const Card = styled.div`
  background-color: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  overflow: hidden;
`;

const CardBody = styled.div`
  padding: 20px;
`;

export { ContainerFluid, Title, Row, Column, Card, CardBody };
