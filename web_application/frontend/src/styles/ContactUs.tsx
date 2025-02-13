/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';

const colors = {
  background: '#2b2b2c',
  surface: '#444e5e',
  accent: '#ff8500',
  accentHover: '#ffa533',
  text: '#ffffff',
  border: '#ff8500',
  error: '#ff4444',
  success: '#4caf50',
  inputBackground: '#363636',
  inputBorder: '#555555',
  labelText: '#cccccc',
  shadowColor: 'rgba(255, 133, 0, 0.15)'
};

export const ContactUsContainer = styled.div`
  max-width: 800px;
  margin: 2rem auto;
  padding: 2rem;
  color: ${colors.text};
  background-color: ${colors.background};
`;

export const ContactHeader = styled.div`
  text-align: center;
  margin-bottom: 3rem;
`;

export const HeaderTitle = styled.h2`
  color: ${colors.accent};
  font-size: 2.8rem;
  margin-bottom: 1.5rem;
  border-bottom: 3px solid ${colors.accent};
  padding-bottom: 1.2rem;
  position: relative;
  
  &:after {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background-color: ${colors.accentHover};
  }
`;

export const HeaderText = styled.p`
  color: ${colors.labelText};
  font-size: 1.2rem;
  line-height: 1.8;
  max-width: 700px;
  margin: 0 auto;
  opacity: 0.9;
`;

export const ContactCard = styled.div`
  background-color: ${colors.surface};
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  overflow: hidden;
  transition: transform 0.3s ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
  }
`;

export const CardBody = styled.div`
  padding: 2rem;
`;

export const FormGroup = styled.div`
  margin-bottom: 2rem;
  position: relative;
`;

export const Label = styled.label`
  display: block;
  color: ${colors.labelText};
  margin-bottom: 0.8rem;
  font-weight: 500;
  font-size: 1.1rem;
  transition: color 0.2s ease;
  
  &:hover {
    color: ${colors.accent};
  }
`;

export const Input = styled.input`
  width: 100%;
  padding: 1rem 1.2rem;
  background-color: ${colors.inputBackground};
  border: 2px solid ${colors.inputBorder};
  border-radius: 8px;
  color: ${colors.text};
  font-size: 1.1rem;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border-color: ${colors.accent};
    box-shadow: 0 0 0 4px ${colors.shadowColor};
    background-color: ${colors.background};
  }

  &:hover:not(:focus) {
    border-color: ${colors.accentHover};
    background-color: ${colors.background};
  }

  &::placeholder {
    color: rgba(255, 255, 255, 0.4);
  }
`;

export const TextArea = styled.textarea`
  width: 100%;
  padding: 1rem 1.2rem;
  background-color: ${colors.inputBackground};
  border: 2px solid ${colors.inputBorder};
  border-radius: 8px;
  color: ${colors.text};
  font-size: 1.1rem;
  min-height: 180px;
  resize: vertical;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border-color: ${colors.accent};
    box-shadow: 0 0 0 4px ${colors.shadowColor};
    background-color: ${colors.background};
  }

  &:hover:not(:focus) {
    border-color: ${colors.accentHover};
    background-color: ${colors.background};
  }

  &::placeholder {
    color: rgba(255, 255, 255, 0.4);
  }
`;

export const SubmitButton = styled.button`
  background-color: ${colors.accent};
  color: ${colors.text};
  padding: 1.2rem 2.5rem;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;

  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transform: translateX(-100%);
  }

  &:hover {
    background-color: ${colors.accentHover};
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(255, 133, 0, 0.3);
    
    &:before {
      transform: translateX(100%);
      transition: transform 0.8s ease;
    }
  }

  &:active {
    transform: translateY(1px);
    box-shadow: 0 2px 8px rgba(255, 133, 0, 0.2);
  }
`;

export const Alert = styled.div`
  padding: 1.2rem;
  margin-bottom: 1.5rem;
  border-radius: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 500;
  animation: slideIn 0.3s ease;

  @keyframes slideIn {
    from {
      transform: translateY(-10px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  &.alert-success {
    background-color: rgba(76, 175, 80, 0.15);
    border: 1px solid ${colors.success};
    color: ${colors.success};
  }

  &.alert-error {
    background-color: rgba(255, 68, 68, 0.15);
    border: 1px solid ${colors.error};
    color: ${colors.error};
  }

  button {
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    font-size: 1.4rem;
    padding: 0 0.5rem;
    opacity: 0.8;
    transition: opacity 0.2s ease;

    &:hover {
      opacity: 1;
    }
  }
`;
