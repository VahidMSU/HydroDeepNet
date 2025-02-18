/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';


export const DashboardContainer = styled.div`
  padding: 2rem;
  background-color: ${colors.background};
  min-height: 100vh;
  color: ${colors.text};
`;

export const DashboardHeader = styled.div`
  text-align: center;
  margin-bottom: 2rem;
  
  h1 {
    color: ${colors.accent};
    font-size: 2.5rem;
    margin-bottom: 1rem;
    border-bottom: 3px solid ${colors.accent};
    padding-bottom: 1rem;
  }
`;

export const ContentGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
`;

export const FileCard = styled.div`
  background-color: ${colors.surface};
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  transition: transform 0.2s ease-in-out;

  &:hover {
    transform: translateY(-4px);
  }
`;

export const FileHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;

  .icon {
    color: ${colors.accent};
    font-size: 1.5rem;
  }

  h3 {
    color: ${colors.text};
    margin: 0;
    font-size: 1.1rem;
  }
`;

export const FileInfo = styled.div`
  margin: 1rem 0;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.7);

  p {
    margin: 0.5rem 0;
  }
`;

export const ActionButton = styled.button`
  background-color: ${colors.accent};
  color: ${colors.text};
  border: none;
  border-radius: 6px;
  padding: 0.8rem 1.2rem;
  font-size: 0.9rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s ease-in-out;

  &:hover {
    background-color: ${colors.accentHover};
    transform: translateY(-2px);
  }

  &:active {
    transform: translateY(0);
  }

  .icon {
    font-size: 1rem;
  }
`;

export const BreadcrumbNav = styled.div`
  margin-bottom: 2rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;

  button {
    background: none;
    border: none;
    color: ${colors.accent};
    cursor: pointer;
    font-size: 1rem;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;

    &:hover {
      background-color: rgba(255, 133, 0, 0.1);
    }
  }

  span {
    color: ${colors.text};
    opacity: 0.7;
  }
`;

export const EmptyState = styled.div`
  text-align: center;
  padding: 3rem;
  color: rgba(255, 255, 255, 0.7);

  .icon {
    font-size: 3rem;
    color: ${colors.accent};
    margin-bottom: 1rem;
  }

  h3 {
    margin-bottom: 1rem;
  }
`;

export const ErrorMessage = styled.div`
  background-color: rgba(255, 68, 68, 0.1);
  border: 1px solid ${colors.error};
  color: ${colors.error};
  padding: 1rem;
  border-radius: 6px;
  margin-bottom: 1rem;
  text-align: center;
`;
