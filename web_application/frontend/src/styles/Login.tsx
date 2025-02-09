/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';

const Container = styled.div`
  margin: 40px auto;
  max-width: 400px;
  padding: 20px;
  background-color: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
`;

const Card = styled.div`
  padding: 20px;
  background-color: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
`;

const CardBody = styled.div`
  padding: 20px;
`;

const Title = styled.h2`
  text-align: center;
  margin-bottom: 20px;
  color: #333;
`;

const Button = styled.a`
  display: block;
  width: 100%;
  padding: 10px;
  margin-bottom: 20px;
  background-color: #28a745;
  color: #fff;
  text-align: center;
  border-radius: 5px;
  text-decoration: none;
  transition: background-color 0.3s;

  &:hover {
    background-color: #218838;
  }
`;

export { Container, Card, CardBody, Title, Button };
