/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';

export const ContactContainer = styled.div`
  max-width: 800px;
  margin: 2rem auto;
  padding: 2rem;
  color: #ffffff;
`;

export const ContactTitle = styled.h2`
  color: white;
  font-size: 4rem;
  margin-bottom: 1.5rem;
  padding-bottom: 1.2rem;
  position: relative;
  text-align: center;
  
  &:after {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background-color: #ffa533;
  }
`;

export const ContentWrapper = styled.div`
  background-color: #444e5e;
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
`;

export const FormGroup = styled.div`
  margin-bottom: 2rem;
  position: relative;
`;

export const Label = styled.label`
  display: block;
  color: #ffffff;
  margin-bottom: 0.8rem;
  font-weight: 500;
  font-size: 1.1rem;
  transition: color 0.2s ease;
  
  &:hover {
    color: #ff8500;
  }
`;

export const Input = styled.input`
  width: 100%;
  padding: 1rem 1.2rem;
  background-color: #333;
  border: 2px solid #555;
  border-radius: 8px;
  color: #ffffff;
  font-size: 1.1rem;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border-color: #ff8500;
    box-shadow: 0 0 0 4px rgba(255, 133, 0, 0.2);
    background-color: #222;
  }

  &:hover:not(:focus) {
    border-color: #ffa533;
    background-color: #222;
  }

  &::placeholder {
    color: rgba(255, 255, 255, 0.4);
  }
`;

export const TextArea = styled.textarea`
  width: 100%;
  padding: 1rem 1.2rem;
  background-color: #333;
  border: 2px solid #555;
  border-radius: 8px;
  color: #ffffff;
  font-size: 1.1rem;
  min-height: 180px;
  resize: vertical;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border-color: #ff8500;
    box-shadow: 0 0 0 4px rgba(255, 133, 0, 0.2);
    background-color: #222;
  }

  &:hover:not(:focus) {
    border-color: #ffa533;
    background-color: #222;
  }

  &::placeholder {
    color: rgba(255, 255, 255, 0.4);
  }
`;

export const SubmitButton = styled.button`
  background-color: #ff8500;
  color: #ffffff;
  padding: 0.6rem 1.2rem;
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
    background-color: #ffa533;
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

interface AlertProps {
  type: 'error' | 'success';
}

export const Alert = styled.div<AlertProps>`
  padding: 1rem;
  margin-bottom: 1rem;
  border-radius: 8px;
  background-color: ${props => props.type === 'error' ? 'rgba(255, 0, 0, 0.1)' : 'rgba(0, 255, 0, 0.1)'};
  color: ${props => props.type === 'error' ? '#ff8500' : '#00ff00'};
  border: 1px solid currentColor;
`;
