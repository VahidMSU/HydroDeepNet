/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';

const Body = styled.body`
  text-size-adjust: 100%;
`;

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 20px auto;
  max-width: 1200px;
  padding: 20px;
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);
  text-align: -webkit-match-parent;
  text-align: match-parent;
`;

export { Body, Container };
