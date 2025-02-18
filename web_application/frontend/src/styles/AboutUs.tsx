import styled from 'styled-components';

export const AboutContainer = styled.div`
  max-width: 800px;
  margin: 2rem auto;
  padding: 2rem;
  color: #ffffff;
`;

export const AboutTitle = styled.h2`
  color: #ff8500;
  font-size: 2.8rem;
  margin-bottom: 1.5rem;
  border-bottom: 3px solid #ff8500;
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

export const SectionHeader = styled.h3`
  color: #ff8500;
  font-size: 1.8rem;
  margin: 2rem 0 1rem;
`;

export const SectionText = styled.p`
  color: #cccccc;
  font-size: 1.2rem;
  line-height: 1.8;
  margin-bottom: 1.5rem;
  opacity: 0.9;
`;

export const ContactSection = styled.div`
  margin-top: 2rem;
  text-align: center;
`;

export const Links = styled.div`
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 1rem;

  a {
    color: #ff8500;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border: 2px solid #ff8500;
    border-radius: 8px;
    transition: all 0.3s ease;

    &:hover {
      background-color: #ff8500;
      color: #ffffff;
    }
  }
`;
