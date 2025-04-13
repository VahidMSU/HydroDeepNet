/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import { keyframes, css } from '@emotion/react';



export const MapControlsContainer = styled.div`
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

export const MapControlButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: white;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
  cursor: pointer;
  transition: all 0.2s ease;
  padding: 0;

  &:hover {
    background: #f0f0f0;
  }

  &.active {
    background: #007bff;
    color: white;
  }
`;



export const StationId = styled.span`
  font-size: 12px;
  color: #666;
  margin-left: 8px;
`;

export const SelectButton = styled.button`
  margin-left: 8px;
  padding: 4px 8px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;

  &:hover {
    background: #0069d9;
  }
`;

export const LoadingOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.7);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 100;
  font-size: 16px;
  color: #333;
`;

export const LoadingIcon = styled.div`
  margin-bottom: 12px;
  font-size: 24px;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`;


// Animations
export const fadeIn = keyframes`
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

export const slideDown = keyframes`
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

export const spin = keyframes`
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
`;

// Main Container - Modified to prevent vertical scrolling
export const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  max-width: 100%;
  overflow: hidden; /* Changed from overflow-x: hidden to prevent all scrolling */
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-color: #1c1c1e;
  color: #ffd380;
  position: fixed; /* Added fixed position */
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
`;

// Header - Adjusted to be more compact
export const Header = styled.header`
  background: linear-gradient(to right, #1e1e1f, #2d2d30);
  color: #ffd380;
  padding: 8px 20px; /* Reduced padding */
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
  z-index: 100;
  width: 100%;
  box-sizing: border-box;
  border-bottom: 1px solid #ff5722;
  flex: 0 0 auto; /* Prevent flexbox from stretching this element */
`;

export const HeaderTitle = styled.h1`
  margin: 0;
  font-size: 1.5rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  color: #ff5722;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: 0.5px;
`;

export const TitleIcon = styled.span`
  margin-right: 12px;
  font-size: 1.4rem;
  color: #ff5722;
`;

// Content Layout - Fixed height calculation
export const Content = styled.div`
  display: flex;
  height: calc(100vh - 50px); /* Adjust to match header height */
  width: 100%;
  overflow: hidden;
  box-sizing: border-box;
  flex: 1 1 auto; /* Allow it to grow/shrink as needed */
  
  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

// Sidebar - Modified to ensure sidebar fits within the available space
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

// ConfigPanel - Modified to ensure proper sizing with fixed container
export const ConfigPanel = styled.div`
  width: 100%;
  height: calc(100% - 45px); /* Subtract the InfoPanel height */
  box-sizing: border-box;
  overflow: hidden;
  border-bottom: 1px solid #3a3a3c;
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
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
  border-left: 4px solid #ff5722;
  
  &:hover {
    background-color: #3a3a3c;
    padding-left: 18px;
  }
`;

export const PanelIcon = styled.span`
  margin-right: 10px;
  color: #ff5722;
  font-size: 1.1rem;
`;

export const ToggleIcon = styled.span<{ isOpen: boolean }>`
  margin-left: auto;
  color: #ff5722;
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

// ConfigPanelContent - Modified to enable scrolling within the panel content while keeping container fixed
export const ConfigPanelContent = styled(PanelContent)`
  flex: 1;
  max-height: calc(100vh - 180px); /* Calculate max height to prevent overflow */
  overflow-y: auto;
  
  /* Custom scrollbar styling */
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: #1e1e1f;
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #3a3a3c;
    border-radius: 4px;
    
    &:hover {
      background: #505050;
    }
  }
  
  /* Ensure scrollbar appears in Firefox */
  scrollbar-width: thin;
  scrollbar-color: #3a3a3c #1e1e1f;
`;

// Map Container - Modified to ensure map fits within available space
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
  color: #ff5722;
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255, 133, 0, 0.3);
`;

export const SectionIcon = styled.span`
  color: #ff5722;
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
    border-color: #ff5722;
    box-shadow: 0 0 0 2px rgba(255, 133, 0, 0.2);
    background-color: #333333;
  }
`;

export const FormSelect = styled.select`
  width: 100%;
  padding: 10px 14px;
  border-radius: 6px;
  border: 1px solid #505050;
  font-size: 1rem;
  background-color: #2b2b2c;
  color: #ffffff;
  box-sizing: border-box;
  transition: all 0.2s ease;
  appearance: none;
  background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%23ffffff%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E");
  background-repeat: no-repeat;
  background-position: right 12px top 50%;
  background-size: 12px auto;
  padding-right: 30px;
  
  &:hover {
    border-color: #687891;
  }
  
  &:focus {
    outline: none;
    border-color: #ff5722;
    box-shadow: 0 0 0 2px rgba(255, 133, 0, 0.2);
    background-color: #333333;
  }
  
  option {
    background-color: #2b2b2c;
    color: #ffffff;
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
  background-color: ${props => props.checked ? '#ff5722' : '#2b2b2c'};
  border: 2px solid ${props => props.checked ? '#ff5722' : '#505050'};
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
export const SubmitButton = styled.button<{ isLoading: boolean, secondary?: boolean }>`
  background-color: ${props => {
    if (props.isLoading) return '#5c5c5e';
    if (props.secondary) return 'transparent';
    return '#ff5722';
  }};
  color: ${props => props.secondary ? '#bbbbbb' : '#1e1e1f'};
  padding: 12px 16px;
  border: ${props => props.secondary ? '1px solid #bbbbbb' : 'none'};
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
  flex: ${props => props.secondary ? '1' : '1'}; /* Give equal flex to both buttons */
  min-width: ${props => props.secondary ? '180px' : '180px'}; /* Ensure minimum width for buttons */
  max-width: ${props => props.secondary ? '45%' : '45%'}; /* Limit maximum width */
  white-space: nowrap; /* Prevent text from wrapping inside button */
  overflow: hidden;
  text-overflow: ellipsis;
  
  &:hover {
    background-color: ${props => {
    if (props.isLoading) return '#5c5c5e';
    if (props.secondary) return 'rgba(255, 255, 255, 0.05)';
    return '#ffa733';
  }};
    transform: ${props => props.isLoading ? 'none' : 'translateY(-2px)'};
    box-shadow: ${props => {
    if (props.isLoading) return 'none';
    if (props.secondary) return 'none';
    return '0 4px 8px rgba(255, 133, 0, 0.3)';
  }};
    border-color: ${props => props.secondary ? '#ffd380' : 'transparent'};
    color: ${props => props.secondary ? '#ffd380' : '#1e1e1f'};
  }
  
  &:active {
    transform: ${props => props.isLoading ? 'none' : 'translateY(0)'};
    box-shadow: ${props => {
    if (props.isLoading) return 'none';
    if (props.secondary) return 'none';
    return '0 2px 4px rgba(255, 133, 0, 0.3)';
  }};
  }

  /* Media query for mobile devices */
  @media (max-width: 768px) {
    min-width: 0; /* Remove min-width on small screens */
    max-width: none; /* Allow full width when needed */
    flex: ${props => props.secondary ? '1 1 100%' : '1 1 100%'}; /* Stack buttons on small screens */
    margin-bottom: 8px;
  }
`;

// Button icon
export const ButtonIcon = styled.span`
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 8px;
  font-size: 1rem;
`;

// Feedback messages - Make feedback messages more compact
export const FeedbackMessage = styled.div<{ type: 'success' | 'error' }>`
  margin-top: 10px;
  padding: 8px 12px;
  border-radius: 8px;
  font-weight: 500;
  animation: ${fadeIn} 0.3s ease-in-out;
  display: flex;
  align-items: center;
  font-size: 0.9rem;
  
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
  border-left: 4px solid #ff5722;
  transition: all 0.3s ease;
  
  &:hover {
    background-color: rgba(255, 133, 0, 0.1);
  }
`;

export const StationName = styled.h4`
  color: #ff5722;
  margin: 0 0 12px 0;
  font-size: 1.1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
  user-select: none;
  transition: all 0.2s ease;

  &:hover {
    color: #ffa533;
  }
`;

export const StationIcon = styled.span`
  color: #ff5722;
  font-size: 1rem;
`;

export const StationInfoContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
  animation: ${fadeIn} 0.3s ease;
  overflow: hidden;
`;

export const StationToggleIcon = styled.span`
  margin-left: auto;
  color: #ff5722;
  transition: transform 0.3s ease, color 0.2s ease;
  
  &:hover {
    color: #ffa533;
  }
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
  background-color: ${props => props.disabled ? '#5c5c5e' : '#ff5722'};
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

// Search results - Adjust search results to fit within fixed container
export const SearchResults = styled.div`
  max-height: 150px; /* Reduced height to prevent overflow */
  overflow-y: auto;
  margin-top: 12px;
  border: 1px solid #505050;
  border-radius: 8px;
  background-color: #2b2b2c;
  animation: ${slideDown} 0.3s ease-out;
  
  /* Custom scrollbar styling */
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #1e1e1f;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #3a3a3c;
    border-radius: 3px;
  }
`;

// Add the missing SearchResultItem component
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

// Tab navigation
export const TabContainer = styled.div`
  display: flex;
  border-bottom: 1px solid #3a3a3c;
  margin-bottom: 16px;
`;

export const TabButton = styled.button<{ active: boolean }>`
  background-color: ${props => props.active ? '#3a3a3c' : 'transparent'};
  color: ${props => props.active ? '#ffd380' : '#bbbbbb'};
  border: none;
  padding: 12px 18px;
  cursor: pointer;
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  font-weight: ${props => props.active ? '600' : '400'};
  transition: all 0.2s ease;
  border-bottom: 2px solid ${props => props.active ? '#ff5722' : 'transparent'};
  
  &:hover {
    background-color: ${props => props.active ? '#3a3a3c' : 'rgba(255, 133, 0, 0.05)'};
    color: #ffd380;
  }
`;

export const TabIcon = styled.span`
  font-size: 1.1rem;
`;

// Step indicator
export const StepIndicator = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid #3a3a3c;
`;

export const StepCircle = styled.div<{ active: boolean, completed: boolean }>`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  background-color: ${props => {
    if (props.completed) return '#4caf50';
    if (props.active) return '#ff5722';
    return '#3a3a3c';
  }};
  color: #1c1c1e;
  transition: all 0.3s ease;
  box-shadow: 0 3px 8px rgba(0, 0, 0, 0.2);
  z-index: 1;
`;

export const StepText = styled.span<{ active: boolean }>`
  color: ${props => props.active ? '#ffd380' : '#bbbbbb'};
  font-weight: ${props => props.active ? '600' : '400'};
  font-size: 0.9rem;
  position: absolute;
  top: 45px;
  transform: translateX(-50%);
  white-space: nowrap;
  transition: all 0.3s ease;
`;

export const StepConnector = styled.div<{ completed: boolean }>`
  height: 3px;
  flex: 1;
  margin: 0 10px;
  background-color: ${props => props.completed ? '#4caf50' : '#3a3a3c'};
  transition: all 0.3s ease;
`;

// Step container
export const StepContainer = styled.div`
  animation: ${fadeIn} 0.3s ease-in-out;
  width: 100%;
`;

// Navigation buttons - Make button margin more compact
export const NavigationButtons = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 12px;
  margin-bottom: 12px;
  gap: 12px;
  flex-wrap: wrap;
  width: 100%;
`;




