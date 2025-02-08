/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import theme from './variables.ts';
import './variables.ts'; // Ensure this path is correct or the file exists

const Body = styled.body`
  text-size-adjust: 100%;
`;

const ContactUsContainer = styled.div`
  max-width: 800px;
  margin: 0 auto;
  padding: ${theme.spacing(2.5)};
`;

const TextCenter = styled.div`
  text-align: center;
  text-align: -webkit-match-parent;
  text-align: match-parent;
`;

const Card = styled.div`
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin-bottom: ${theme.spacing(2.5)};
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);
`;

const CardBody = styled.div`
  padding: ${theme.spacing(2.5)};
`;

export { Body, ContactUsContainer, TextCenter, Card, CardBody };
