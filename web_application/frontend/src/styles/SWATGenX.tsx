import styled, { keyframes, css } from 'styled-components';

// Animations
const fadeIn = keyframes`
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

const slideDown = keyframes`
  from {
    opacity: 0;
    transform: translateY(-10px);
    max-height: 0;
  }
  to {
    opacity: 1;
    transform: translateY(0);
    max-height: 250px;
  }
`;

const spin = keyframes`
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
`;

// Main Container
export const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-color: #1c1c1e;
  color: #ffd380;
`;

// Header
export const Header = styled.header`
  background: linear-gradient(to right, #1e1e1f, #2d2d30);
  color: #ffd380;
  padding: 12px 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
  z-index: 100;
  width: 100%;
  box-sizing: border-box;
  border-bottom: 1px solid #ff8500;
`;

export const HeaderTitle = styled.h1`
  margin: 0;
  font-size: 1.5rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  color: #ff8500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: 0.5px;
`;

export const TitleIcon = styled.span`
  margin-right: 12px;
  font-size: 1.4rem;
  color: #ff8500;
`;

// Content Layout
export const Content = styled.div`
  display: flex;
  height: calc(100vh - 60px);
  width: 100%;
  overflow: hidden;
  box-sizing: border-box;
  
  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

// Sidebar
export const Sidebar = styled.div`
  width: 360px;
  min-width: 300px;
  max-width: 40%;
  background-color: #2b2b2c;
  border-right: 1px solid #3a3a3c;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  height: 100%;
  flex-shrink: 0;
  box-sizing: border-box;
  box-shadow: 2px 0 10px rgba(0, 0, 0, 0.2);
  
  @media (max-width: 768px) {
    width: 100%;
    max-width: 100%;
    height: 50%;
    border-right: none;
    border-bottom: 1px solid #3a3a3c;
  }
`;

export const InfoPanel = styled.div`
  width: 100%;
  box-sizing: border-box;
  overflow: hidden;
  border-bottom: 1px solid #3a3a3c;
  display: flex;
  flex-direction: column;
  flex: 0 0 auto;
`;

export const ConfigPanel = styled.div`
  width: 100%;
  box-sizing: border-box;
  overflow: hidden;
  border-bottom: 1px solid #3a3a3c;
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  overflow: hidden;
`;

// Panel header
export const PanelHeader = styled.div`
  display: flex;
  align-items: center;
  padding: 14px 16px;
  background-color: #2b2b2c;
  cursor: pointer;
  font-weight: 600;
  color: #ffd380;
  transition: all 0.2s ease;
  border-left: 4px solid #ff8500;
  
  &:hover {
    background-color: #3a3a3c;
    padding-left: 18px;
  }
`;

export const PanelIcon = styled.span`
  margin-right: 10px;
  color: #ff8500;
  font-size: 1.1rem;
`;

export const ToggleIcon = styled.span<{ isOpen: boolean }>`
  margin-left: auto;
  color: #ff8500;
  transition: transform 0.3s ease;
  transform: ${props => props.isOpen ? 'rotate(180deg)' : 'rotate(0)'};
`;

// Panel content
export const PanelContent = styled.div`
  padding: 16px;
  overflow-y: auto;
  overflow-x: hidden;
  color: #bbbbbb;
  box-sizing: border-box;
  width: 100%;
  background-color: #2b2b2c;
  line-height: 1.5;
  
  p {
    margin-bottom: 12px;
    color: #bbbbbb;
  }
  
  ul {
    padding-left: 18px;
    margin-bottom: 12px;
    color: #bbbbbb;
  }
  
  li {
    margin-bottom: 6px;
  }
  
  strong {
    color: #ffd380;
    font-weight: 600;
  }
  
  @media (max-width: 480px) {
    padding: 10px;
  }
`;

export const ConfigPanelContent = styled(PanelContent)`
  flex: 1;
  overflow-y: auto;
`;

// Map Container
export const MapContainer = styled.div`
  flex: 1;
  overflow: hidden;
  position: relative;
  background-color: #1c1c1e;
  padding: 16px;
  box-sizing: border-box;
  min-width: 0;
  
  @media (max-width: 768px) {
    height: 50%;
    width: 100%;
  }
`;

export const MapInnerContainer = styled.div`
  height: 100%;
  width: 100%;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
  box-sizing: border-box;
  border: 1px solid #3a3a3c;
  position: relative;
`;

// Form Styles
export const ModelSettingsFormContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 18px;
  width: 100%;
  box-sizing: border-box;
  overflow-x: hidden;
`;

export const FormSection = styled.div`
  background-color: #3a3a3c;
  border-radius: 10px;
  padding: 18px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
  transition: all 0.25s ease;
  width: 100%;
  box-sizing: border-box;
  overflow: hidden;
  border: 1px solid #505050;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    border-color: #687891;
  }
  
  @media (max-width: 768px) {
    padding: 14px;
  }
  
  @media (max-width: 480px) {
    padding: 10px;
  }
`;

export const SectionTitle = styled.h3`
  font-size: 1.05rem;
  margin: 0 0 14px 0;
  color: #ff8500;
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255, 133, 0, 0.3);
`;

export const SectionIcon = styled.span`
  color: #ff8500;
`;

export const InputGroup = styled.div`
  margin-bottom: 14px;
  
  @media (max-width: 768px) {
    margin-bottom: 10px;
  }
`;

export const InputLabel = styled.label`
  display: block;
  margin-bottom: 8px;
  font-size: 0.9rem;
  color: #bbbbbb;
  font-weight: 500;
`;

export const FormInput = styled.input`
  width: 100%;
  padding: 10px 14px;
  border-radius: 6px;
  border: 1px solid #505050;
  font-size: 1rem;
  background-color: #2b2b2c;
  color: #ffffff;
  box-sizing: border-box;
  transition: all 0.2s ease;
  
  &:hover {
    border-color: #687891;
  }
  
  &:focus {
    outline: none;
    border-color: #ff8500;
    box-shadow: 0 0 0 2px rgba(255, 133, 0, 0.2);
    background-color: #333333;
  }
`;

export const CheckboxGroup = styled.div`
  margin-bottom: 10px;
  display: flex;
  align-items: center;
`;

export const CheckboxLabel = styled.label`
  display: flex;
  align-items: center;
  cursor: pointer;
  font-size: 0.95rem;
  color: #bbbbbb;
  padding: 6px 0;
  transition: color 0.2s;
  
  &:hover {
    color: #ffd380;
  }
`;

export const CheckboxInput = styled.input`
  height: 0;
  width: 0;
  opacity: 0;
  position: absolute;
`;

export const CheckboxCustom = styled.div<{ checked: boolean }>`
  width: 20px;
  height: 20px;
  background-color: ${props => props.checked ? '#ff8500' : '#2b2b2c'};
  border: 2px solid ${props => props.checked ? '#ff8500' : '#505050'};
  border-radius: 4px;
  margin-right: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  position: relative;
  flex-shrink: 0;
  
  &:after {
    content: '';
    position: absolute;
    display: ${props => props.checked ? 'block' : 'none'};
    width: 5px;
    height: 10px;
    border: solid white;
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
    top: 2px;
  }
`;

export const CheckboxText = styled.span`
  margin-left: 8px;
`;

// Button styling
export const SubmitButton = styled.button<{ isLoading: boolean }>`
  background-color: ${props => props.isLoading ? '#5c5c5e' : '#ff8500'};
  color: #1e1e1f;
  padding: 12px 16px;
  border: none;
  border-radius: 8px;
  cursor: ${props => props.isLoading ? 'not-allowed' : 'pointer'};
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  transition: all 0.25s ease;
  margin-top: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-size: 0.95rem;
  
  &:hover {
    background-color: ${props => props.isLoading ? '#5c5c5e' : '#ffa733'};
    transform: ${props => props.isLoading ? 'none' : 'translateY(-2px)'};
    box-shadow: ${props => props.isLoading ? 'none' : '0 4px 8px rgba(255, 133, 0, 0.3)'};
  }
  
  &:active {
    transform: ${props => props.isLoading ? 'none' : 'translateY(0)'};
    box-shadow: ${props => props.isLoading ? 'none' : '0 2px 4px rgba(255, 133, 0, 0.3)'};
  }
`;

export const ButtonIcon = styled.span`
  font-size: 1.1rem;
`;

// Feedback messages
export const FeedbackMessage = styled.div<{ type: 'success' | 'error' }>`
  margin-top: 16px;
  padding: 12px 14px;
  border-radius: 8px;
  font-weight: 500;
  animation: ${fadeIn} 0.3s ease-in-out;
  display: flex;
  align-items: center;
  
  ${props => props.type === 'success' && css`
    background-color: rgba(76, 175, 80, 0.15);
    color: #4caf50;
    border: 1px solid rgba(76, 175, 80, 0.3);
  `}
  
  ${props => props.type === 'error' && css`
    background-color: rgba(244, 67, 54, 0.15);
    color: #f44336;
    border: 1px solid rgba(244, 67, 54, 0.3);
  `}
`;

export const FeedbackIcon = styled.span`
  margin-right: 10px;
  font-size: 1.2rem;
`;

// Station details
export const StationDetailsContainer = styled.div`
  background-color: rgba(255, 133, 0, 0.05);
  border-radius: 8px;
  padding: 16px;
  margin-top: 16px;
  border-left: 4px solid #ff8500;
`;

export const StationName = styled.h4`
  color: #ff8500;
  margin: 0 0 12px 0;
  font-size: 1.1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
`;

export const StationIcon = styled.span`
  color: #ff8500;
  font-size: 1rem;
`;

export const StationInfoContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

export const InfoItem = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 12px;
`;

export const InfoIcon = styled.span`
  color: #bbbbbb;
  margin-top: 4px;
  font-size: 0.9rem;
`;

export const InfoContent = styled.div`
  display: flex;
  flex-direction: column;
`;

export const InfoLabel = styled.label`
  font-size: 0.8rem;
  color: #9e9e9e;
  margin-bottom: 3px;
`;

export const InfoValue = styled.span`
  color: #ffd380;
  font-size: 0.95rem;
`;

// Search form
export const SearchForm = styled.div`
  margin-bottom: 16px;
`;

export const SearchInputGroup = styled.div`
  margin-bottom: 10px;
`;

export const SearchInputWrapper = styled.div`
  display: flex;
  gap: 8px;
`;

export const SearchButton = styled.button<{ disabled?: boolean }>`
  background-color: ${props => props.disabled ? '#5c5c5e' : '#ff8500'};
  color: #1e1e1f;
  border: none;
  border-radius: 6px;
  padding: 0 15px;
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  
  &:hover {
    background-color: ${props => props.disabled ? '#5c5c5e' : '#ffa733'};
    box-shadow: ${props => props.disabled ? 'none' : '0 2px 6px rgba(255, 133, 0, 0.3)'};
  }
  
  &:active {
    background-color: ${props => props.disabled ? '#5c5c5e' : '#e06e00'};
  }
`;

// Search results
export const SearchResults = styled.div`
  max-height: 250px;
  overflow-y: auto;
  margin-top: 12px;
  border: 1px solid #505050;
  border-radius: 8px;
  background-color: #2b2b2c;
  animation: ${slideDown} 0.3s ease-out;
`;

export const SearchResultItem = styled.div`
  padding: 12px 14px;
  border-bottom: 1px solid #3a3a3c;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:last-child {
    border-bottom: none;
  }
  
  &:hover {
    background-color: rgba(255, 133, 0, 0.1);
    padding-left: 18px;
  }
  
  strong {
    display: block;
    color: #ffd380;
    font-size: 0.95rem;
    margin-bottom: 3px;
  }
  
  span {
    display: block;
    color: #9e9e9e;
    font-size: 0.85rem;
  }
`;

// Loading spinner
export const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: #ffffff;
  animation: ${spin} 1s ease-in-out infinite;
  margin-right: 8px;
`;




