/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
// Updated dark theme color palette
import colors from './colors.tsx';

export const FieldText = colors.FieldText;

// New component for titles with black font
export const HeaderTitle = styled.h1`
  text-align: center;
  margin-bottom: 30px;
  color: #333;
  font-size: 2.5rem;
  font-weight: bold;
`;

export const Row = styled.div`
  display: flex;
  flex-wrap: wrap;
  width: 100%;
  height: 100%;
  gap: 1.5rem;
  margin: 0;
  padding: 0;
`;

interface ColumnProps {
  width?: number;
  minWidth?: string;
  mobileMinWidth?: string;
}

export const Column = styled.div<ColumnProps>`
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

export const SectionTitle = styled.h3`
  color: ${colors.accentAlt};   // Use new accentAlt for titles
  font-size: 1.3rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  padding-bottom: 0.8rem;
  border-bottom: 2px solid ${colors.accentAlt};
  display: flex;
  align-items: center;
  gap: 0.8rem;
  letter-spacing: 0.5px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);

  .icon {
    font-size: 1.2rem;
    color: ${colors.accentAlt};
  }
`;

export const CardBody = styled.div`
  padding: 0.75rem;
`;

export const ContainerFluid = styled.div`
  padding: 15px 25px 15px 15px;
  background-color: ${colors.background};
  background-image: none; // Removed gradient to keep it solid grey
  min-height: 100vh;
  max-width: 100vw;
`;

export const Content = styled.main`
  margin-left: 65px;
  padding: 0 20px 0 0;
  display: flex;
  flex-wrap: nowrap;  // Changed from "wrap" to "nowrap" to keep columns fixed in place
  width: calc(100vw - 85px);
  height: calc(100vh - 30px);
  overflow: hidden;
  box-sizing: border-box;

  @media (max-width: 768px) {
    margin-left: 0;
    width: 100vw;
    padding: 0 10px;
  }
`;

export const ContentWrapper = styled.div`
  flex-grow: 1;
  margin-left: 270px;
  padding: 2.5rem;
  background-color: ${colors.surface};
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
  overflow-y: auto;
  border: 1px solid ${colors.border};
  transition: all 0.4s;

  @media (max-width: 768px) {
    margin-left: 0;
    height: auto;
    max-height: none;
  }
`;

export const MapWrapper = styled(ContentWrapper)`
  width: 100%;
  margin: 0;
  padding: 1rem 1.5rem 1rem 1rem;
  position: sticky;
  top: 0;
  overflow: hidden;
  height: 100%;   // Ensure the map container fills its parent height
  @media (max-width: 1400px) {
    height: auto !important;
  }
`;

export const DescriptionContainer = styled.div`
  background-color: #444e5e;
  padding: 24px;
  margin: 20px 0;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  color: white;
  margin-bottom: 20px;
`;

export const InfoBox = styled.div`
  background-color: ${colors.surfaceLight};
  border: 1px solid ${colors.border};
  border-radius: 12px;
  padding: 1.5rem;
  margin-top: 1rem;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);

  label {
    color: #000000; // Force characteristics text to black
  }

  p {
    color: #000000; // Force description/characteristics results to black
    margin-bottom: 1rem;
    line-height: 1.6;
  }
`;

interface DescriptionHeaderProps {
  isOpen?: boolean;
}

export const DescriptionHeader = styled.div<DescriptionHeaderProps>`
  cursor: pointer;
  padding: 12px 16px;
  background-color: ${colors.headerBg};
  color: ${colors.textSecondary};
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-radius: 8px;
  margin-bottom: ${props => props.isOpen ? '10px' : '0'};
  border: 1px solid ${colors.border};
  
  &:hover {
    background-color: ${colors.surfaceLight};
    color: ${colors.text};
  }
`;

export const StrongText = styled.span`
  color: black;
  font-weight: bold;
`;

export const TitleText = styled.h4`
  color: ${colors.TitleText};
  font-weight: 600;
  margin-bottom: 0.5rem;
`;


export const ListElement = styled.ul`
  color: white;
  list-style-type: disc;
  margin-left: 1.5rem;
  
  li {
    margin-bottom: 0.5rem;
  }
`;

export const Section = styled.section`
  color: ${colors.SectionText};
  width: 100%;
  height: 100%;
`;

export const ModelContainer = styled.div`
  background-color: #444e5e;
  padding: 24px;
  margin: 20px 0;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  color: white;
  margin-bottom: 20px;
`;

export const StyledInput = styled.input`
  width: 100%;
  padding: 12px;
  margin: 8px 0;
  border: 1px solid ${colors.border};
  border-radius: 8px;
  background-color: ${colors.inputBg};
  color: ${colors.FieldText};
  font-size: 1rem;
  transition: all 0.3s ease;

  &:focus {
    border-color: ${colors.borderAccent};
    outline: none;
    box-shadow: 0 0 0 2px ${colors.borderAccent}33;
  }

  &::placeholder {
    color: #000000; // Force placeholder text to be black
  }

  &[type="checkbox"] {
    width: auto;
    margin-right: 8px;
  }
`;

export const StyledButton = styled.button`
  background-color: ${colors.accent};
  color: ${colors.text};
  padding: 12px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.3s ease;
  text-transform: uppercase;
  letter-spacing: 0.5px;

  &:hover {
    background-color: ${colors.accentHover};
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  &:active {
    transform: translateY(0);
  }

  &:disabled {
    background-color: ${colors.border};
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

export const SearchResults = styled.ul`
  list-style: none;
  padding: 0;
  margin: 1rem 0;
  background-color: ${colors.surfaceLight};
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid ${colors.border};
`;

export const SearchResultItem = styled.li`
  padding: 0.75rem 1rem;
  margin: 0;
  cursor: pointer;
  color: ${colors.textSecondary};
  border-bottom: 1px solid ${colors.border};
  transition: all 0.2s ease;

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    background-color: ${colors.surface};
    color: ${colors.text};
  }

  strong {
    color: ${colors.accent};
    margin-right: 0.5rem;
    font-weight: 500;
  }
`;

export const Label = styled.label`
  color: ${colors.textSecondary};
  margin-bottom: 0.5rem;
  display: block;
  font-weight: 500;
  font-size: 0.95rem;
  letter-spacing: 0.3px;
`;
