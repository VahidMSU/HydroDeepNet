import styled from 'styled-components';

export const AboutUsContainer = styled.div`
  margin: 40px;
  padding: 20px;
  background-color: #f9f9f9;
  color: #333;
  line-height: 1.6;
`;

export const Card = styled.div`
  background: white;
  padding: 20px;
  border-radius: 5px;
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
`;

export const CardBody = styled.div`
  padding: 20px;
`;

export const Header = styled.h2`
  margin-bottom: 20px;
  font-size: 1.5rem;
  color: #333;
`;

export const SubHeader = styled.h3`
  margin-bottom: 15px;
  font-size: 1.25rem;
  color: #555;
`;

export const Paragraph = styled.p`
  margin-bottom: 15px;
  font-size: 1rem;
  color: #666;
  line-height: 1.5;
`;

export const List = styled.ol`
  margin-bottom: 15px;
  padding-left: 20px;

  li {
    margin-bottom: 10px;
  }
`;

export const Link = styled.a`
  color: #007bff;
  text-decoration: none;

  &:hover {
    text-decoration: underline;
  }
`;
