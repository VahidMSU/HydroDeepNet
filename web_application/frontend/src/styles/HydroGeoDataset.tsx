/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';

export const HydroGeoContainer = styled.div`
  display: flex;
  flex-direction: column;
  padding: 2rem;
  background-color: ${colors.background};
  min-height: 100vh;
  color: ${colors.text};
`;

export const HydroGeoHeader = styled.div`
  text-align: center;
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, ${colors.surfaceDark} 0%, ${colors.surface} 100%);
  border-radius: 12px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
  border-bottom: 3px solid ${colors.accent};
  
  h1 {
    color: ${colors.accent};
    font-size: 2.5rem;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
  }
  
  p {
    font-size: 1.1rem;
    color: ${colors.textSecondary};
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
  }
`;

export const ContentLayout = styled.div`
  display: flex;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
  
  @media (max-width: 992px) {
    flex-direction: column;
  }
`;

export const QuerySidebar = styled.div`
  width: 30%;
  min-width: 350px;
  background: linear-gradient(160deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 12px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  border-left: 4px solid ${colors.accent};
  
  @media (max-width: 992px) {
    width: 100%;
  }
`;

export const MapContainer = styled.div`
  flex: 1;
  min-height: 600px;
  background-color: ${colors.surfaceDark};
  border-radius: 12px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  overflow: hidden;
  position: relative;
  border: 1px solid ${colors.border};
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(to right, ${colors.info}, ${colors.accent});
  }
  
  /* Add additional styles to ensure proper sizing */
  display: flex;
  flex-direction: column;
  
  /* Ensure the inner content takes full height */
  & > div {
    flex: 1;
    min-height: 400px;
  }
`;

export const QuerySidebarHeader = styled.div`
  background-color: ${colors.surfaceDark};
  padding: 1.2rem;
  border-bottom: 1px solid ${colors.border};
  
  h2 {
    color: ${colors.accent};
    margin: 0;
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    
    .icon {
      font-size: 1.3rem;
    }
  }
`;

export const QuerySidebarContent = styled.div`
  padding: 1.5rem;
  flex: 1;
  overflow-y: auto;
`;

export const FormGroup = styled.div`
  margin-bottom: 1.8rem;
  
  label {
    display: block;
    color: ${colors.textSecondary};
    margin-bottom: 0.6rem;
    font-size: 0.95rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    .icon {
      color: ${colors.accent};
      font-size: 1rem;
    }
  }
`;

export const InputField = styled.div`
  position: relative;
  
  select, input {
    width: 100%;
    padding: 0.8rem 1rem;
    background-color: ${colors.inputBg};
    border: 1px solid ${colors.border};
    border-radius: 8px;
    color: ${colors.inputText};
    font-size: 1rem;
    transition: all 0.2s ease;
    appearance: none;
    
    &:focus {
      outline: none;
      border-color: ${colors.accent};
      box-shadow: 0 0 0 2px rgba(255, 133, 0, 0.2);
    }
    
    &:hover {
      border-color: ${colors.borderLight};
    }
    
    &::placeholder {
      color: ${colors.textMuted};
    }
  }
  
  select {
    padding-right: 2rem;
  }
  
  .select-arrow {
    position: absolute;
    right: 1rem;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
    color: ${colors.accent};
  }
`;

export const SubmitButton = styled.button`
  width: 100%;
  background: linear-gradient(135deg, ${colors.accent} 0%, #e67300 100%);
  color: ${colors.textInverse};
  border: none;
  border-radius: 8px;
  padding: 1rem;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.8rem;
  transition: all 0.2s ease-in-out;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  box-shadow: 0 4px 10px rgba(255, 133, 0, 0.3);
  
  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(255, 133, 0, 0.4);
  }
  
  &:active:not(:disabled) {
    transform: translateY(0);
  }
  
  &:disabled {
    background: ${colors.disabled};
    cursor: not-allowed;
    box-shadow: none;
  }
  
  .icon {
    font-size: 1.2rem;
  }
`;

export const InfoCard = styled.div`
  background-color: ${colors.surface};
  border-radius: 10px;
  padding: 1.2rem;
  margin-bottom: 1.5rem;
  border-left: 4px solid ${colors.info};
  
  h3 {
    color: ${colors.text};
    margin-top: 0;
    margin-bottom: 0.8rem;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    .icon {
      color: ${colors.info};
    }
  }
  
  p {
    color: ${colors.textSecondary};
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.5;
  }
`;

export const ResultsContainer = styled.div`
  background: linear-gradient(160deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 12px;
  padding: 1.5rem;
  margin-top: 1.5rem;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  
  h2 {
    color: ${colors.accent};
    margin-top: 0;
    margin-bottom: 1.2rem;
    font-size: 1.5rem;
    border-bottom: 1px solid ${colors.border};
    padding-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    
    .icon {
      color: ${colors.accent};
    }
  }
  
  pre {
    background-color: ${colors.surfaceDark};
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
    color: ${colors.textSecondary};
    border: 1px solid ${colors.border};
    font-family: 'Courier New', Courier, monospace;
  }
`;

export const ChatContainer = styled.div`
  background: linear-gradient(160deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 12px;
  overflow: hidden;
  margin-top: 1.5rem;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
`;

export const ChatHeader = styled.div`
  background-color: ${colors.surfaceDark};
  padding: 1.2rem;
  border-bottom: 1px solid ${colors.border};
  display: flex;
  align-items: center;
  justify-content: center;
  
  h2 {
    color: ${colors.accent};
    margin: 0;
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    
    .icon {
      color: ${colors.accent};
      font-size: 1.3rem;
    }
  }
`;

export const ChatMessagesContainer = styled.div`
  height: 350px;
  padding: 1.5rem;
  overflow-y: auto;
  background-color: rgba(0, 0, 0, 0.2);
  
  /* Custom scrollbar for chat */
  &::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: ${colors.surfaceDark};
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: ${colors.border};
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: ${colors.accent};
  }
`;

export const MessageBubble = styled.div`
  max-width: 80%;
  margin-bottom: 1rem;
  padding: 1rem;
  border-radius: 12px;
  position: relative;
  
  &.user {
    background-color: ${colors.accent};
    color: ${colors.textInverse};
    align-self: flex-end;
    margin-left: auto;
    border-bottom-right-radius: 2px;
  }
  
  &.bot {
    background-color: ${colors.surface};
    color: ${colors.textSecondary};
    align-self: flex-start;
    margin-right: auto;
    border-bottom-left-radius: 2px;
  }
`;

export const MessageList = styled.div`
  display: flex;
  flex-direction: column;
`;

export const ChatInputContainer = styled.form`
  display: flex;
  padding: 1rem;
  background-color: ${colors.surfaceDark};
  border-top: 1px solid ${colors.border};
  gap: 0.8rem;
  
  input {
    flex: 1;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    border: 1px solid ${colors.border};
    background-color: ${colors.inputBg};
    color: ${colors.inputText};
    font-size: 1rem;
    
    &:focus {
      outline: none;
      border-color: ${colors.accent};
    }
  }
  
  button {
    background-color: ${colors.accent};
    color: ${colors.textInverse};
    border: none;
    border-radius: 8px;
    padding: 0 1.2rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
    
    &:hover:not(:disabled) {
      background-color: ${colors.accentHover};
    }
    
    &:disabled {
      background-color: ${colors.disabled};
      cursor: not-allowed;
    }
    
    .icon {
      font-size: 1.2rem;
    }
  }
`;

export const ThinkingIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: ${colors.textSecondary};
  margin-bottom: 1rem;
  background-color: ${colors.surface};
  padding: 1rem;
  border-radius: 12px;
  max-width: 150px;
  
  .dot {
    width: 8px;
    height: 8px;
    background-color: ${colors.accent};
    border-radius: 50%;
    animation: pulse 1.5s infinite ease-in-out;
    
    &:nth-of-type(1) {
      animation-delay: 0s;
    }
    
    &:nth-of-type(2) {
      animation-delay: 0.2s;
    }
    
    &:nth-of-type(3) {
      animation-delay: 0.4s;
    }
  }
  
  @keyframes pulse {
    0% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.2);
      opacity: 0.5;
    }
    100% {
      transform: scale(1);
      opacity: 1;
    }
  }
`;

export const CoordinatesDisplay = styled.div`
  background-color: ${colors.surfaceDark};
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  border: 1px solid ${colors.border};
  
  .title {
    color: ${colors.accent};
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    .icon {
      color: ${colors.accent};
    }
  }
  
  .value {
    color: ${colors.text};
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.9rem;
    overflow-x: auto;
    white-space: nowrap;
  }
`;

export const TabContainer = styled.div`
  margin-top: 1.5rem;
`;

export const TabNav = styled.div`
  display: flex;
  border-bottom: 2px solid ${colors.border};
`;

export const TabButton = styled.button`
  padding: 1rem 1.5rem;
  background: transparent;
  border: none;
  color: ${colors.textSecondary};
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
  
  &:hover {
    color: ${colors.accent};
  }
  
  &.active {
    color: ${colors.accent};
    
    &:after {
      content: '';
      position: absolute;
      bottom: -2px;
      left: 0;
      right: 0;
      height: 3px;
      background-color: ${colors.accent};
    }
  }
`;

export const TabContent = styled.div`
  padding: 1.5rem 0;
  display: none;
  
  &.active {
    display: block;
    animation: fadeIn 0.3s ease-in-out;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
`;

export const ReportForm = styled.form`
  background: linear-gradient(160deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 12px;
  padding: 1.5rem;
  margin-top: 1.5rem;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
`;

export const ReportFormHeader = styled.div`
  margin-bottom: 1.5rem;
  
  h3 {
    color: ${colors.accent};
    margin-top: 0;
    font-size: 1.4rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    
    .icon {
      color: ${colors.accent};
    }
  }
  
  p {
    color: ${colors.textSecondary};
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.5;
  }
`;

export const ReportRow = styled.div`
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  
  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

export const ReportStatusContainer = styled.div`
  margin-top: 1.5rem;
  
  h4 {
    color: ${colors.accent};
    margin-top: 0;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    .icon {
      color: ${colors.accent};
    }
  }
`;

export const ReportList = styled.div`
  margin-top: 1rem;
`;

export const ReportItem = styled.div`
  background-color: ${colors.surface};
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  border-left: 4px solid ${colors.accent};
  
  &.processing {
    border-left-color: ${colors.info};
  }
  
  &.failed {
    border-left-color: ${colors.error};
  }
  
  .report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    
    .report-title {
      font-weight: 600;
      color: ${colors.text};
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .report-date {
      color: ${colors.textMuted};
      font-size: 0.9rem;
    }
  }
  
  .report-details {
    color: ${colors.textSecondary};
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
  }
  
  .report-actions {
    display: flex;
    gap: 0.8rem;
    margin-top: 0.8rem;
    
    button {
      background-color: ${colors.surfaceDark};
      color: ${colors.text};
      border: 1px solid ${colors.border};
      border-radius: 4px;
      padding: 0.5rem 0.8rem;
      font-size: 0.9rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      cursor: pointer;
      transition: all 0.2s;
      
      &:hover {
        background-color: ${colors.accent};
        color: ${colors.textInverse};
      }
      
      .icon {
        font-size: 0.9rem;
      }
    }
  }
`;

export const ReportProgressBar = styled.div`
  height: 6px;
  background-color: ${colors.surfaceDark};
  border-radius: 3px;
  margin-top: 0.5rem;
  overflow: hidden;
  
  .progress-inner {
    height: 100%;
    background: linear-gradient(to right, ${colors.info}, ${colors.accent});
    width: 0%;
    transition: width 0.3s ease;
    animation: progress-animation 1.5s infinite;
  }
  
  @keyframes progress-animation {
    0% {
      background-position: 0% 50%;
    }
    50% {
      background-position: 100% 50%;
    }
    100% {
      background-position: 0% 50%;
    }
  }
`;

export const CheckboxContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
  
  label {
    color: ${colors.textSecondary};
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    .icon {
      color: ${colors.accent};
    }
  }
  
  input[type="checkbox"] {
    width: 18px;
    height: 18px;
    cursor: pointer;
  }
`;
