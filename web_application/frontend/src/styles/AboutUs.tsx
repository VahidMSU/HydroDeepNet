/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';

export const AboutUsContainer = styled.div`
  padding: 3rem 2rem;
  background-color: ${colors.background};
  min-height: 100vh;
  color: ${colors.text};
  max-width: 1200px;
  margin: 0 auto;
`;

export const AboutUsHeader = styled.div`
  text-align: center;
  margin-bottom: 3rem;
  position: relative;
  
  h1 {
    color: ${colors.accent};
    font-size: 3rem;
    margin: 0;
    padding-bottom: 1.2rem;
    letter-spacing: 1px;
    position: relative;
    display: inline-block;
    
    &:after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 10%;
      right: 10%;
      height: 4px;
      background: linear-gradient(to right, transparent, ${colors.accent}, transparent);
    }
  }
  
  p {
    margin-top: 1.2rem;
    color: ${colors.textSecondary};
    font-size: 1.2rem;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
  }
`;

export const AccordionContainer = styled.div`
  margin-bottom: 2.5rem;
`;

export const AccordionItem = styled.div`
  background: linear-gradient(135deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 1.2rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  border: 1px solid ${colors.border};
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    border-color: ${colors.accent};
  }
`;

export const AccordionHeader = styled.div<{ isOpen?: boolean }>`
  background: ${props => props.isOpen 
    ? `linear-gradient(135deg, ${colors.accent} 0%, #e67300 100%)`
    : `linear-gradient(135deg, ${colors.surfaceLight} 0%, ${colors.surface} 100%)`
  };
  padding: 1.2rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  transition: all 0.3s ease;
  border-left: 5px solid ${props => props.isOpen ? colors.accent : 'transparent'};

  &:hover {
    background: ${props => props.isOpen 
      ? `linear-gradient(135deg, ${colors.accent} 0%, #e67300 100%)`
      : `linear-gradient(135deg, ${colors.surfaceLight} 30%, ${colors.surface} 100%)`
    };
  }
  
  h3 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 600;
    color: ${props => props.isOpen ? colors.textInverse : colors.accent};
    display: flex;
    align-items: center;
    gap: 0.8rem;
  }
  
  .icon {
    font-size: 1.3rem;
    transition: transform 0.3s ease;
    transform: ${props => props.isOpen ? 'rotate(180deg)' : 'rotate(0)'};
    color: ${props => props.isOpen ? colors.textInverse : colors.accent};
  }
`;

export const AccordionContent = styled.div<{ isOpen?: boolean }>`
  padding: ${props => props.isOpen ? '1.5rem 2rem' : '0 2rem'};
  max-height: ${props => props.isOpen ? '700px' : '0'};
  overflow: hidden;
  transition: all 0.5s ease;
  opacity: ${props => props.isOpen ? 1 : 0};
  
  p {
    color: ${colors.textSecondary};
    font-size: 1.1rem;
    line-height: 1.8;
    margin: 0 0 1rem 0;
  }
  
  ul {
    padding-left: 1.5rem;
    color: ${colors.textSecondary};
    
    li {
      margin-bottom: 0.8rem;
      line-height: 1.5;
    }
  }
`;

export const FeatureCard = styled.div`
  background: linear-gradient(135deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
  display: flex;
  border-left: 4px solid ${colors.accent};
  
  &:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
  }
  
  .icon-container {
    width: 4rem;
    height: 4rem;
    background-color: rgba(255, 133, 0, 0.15);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1.5rem;
    flex-shrink: 0;
    
    .icon {
      color: ${colors.accent};
      font-size: 2rem;
    }
  }
  
  .content {
    flex: 1;
    
    h3 {
      margin: 0 0 0.8rem 0;
      font-size: 1.3rem;
      color: ${colors.accent};
    }
    
    p {
      margin: 0;
      color: ${colors.textSecondary};
      font-size: 1rem;
      line-height: 1.6;
    }
  }
`;

export const ContactSection = styled.div`
  background: linear-gradient(135deg, ${colors.surfaceLight} 0%, ${colors.surface} 100%);
  border-radius: 10px;
  padding: 2.5rem 2rem;
  margin-top: 3rem;
  text-align: center;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  border-top: 4px solid ${colors.accent};
  
  h2 {
    color: ${colors.accent};
    margin: 0 0 1.5rem 0;
    font-size: 2rem;
    position: relative;
    display: inline-block;
    padding-bottom: 0.8rem;
    
    &:after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 20%;
      right: 20%;
      height: 3px;
      background: linear-gradient(to right, transparent, ${colors.accent}, transparent);
    }
  }
  
  p {
    color: ${colors.textSecondary};
    font-size: 1.1rem;
    margin-bottom: 2rem;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.7;
  }
`;

export const ButtonContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 1.2rem;
  margin-top: 2rem;
`;

export const AboutUsButton = styled.a`
  background: linear-gradient(135deg, ${colors.accent} 0%, #e67300 100%);
  color: ${colors.textInverse};
  padding: 0.9rem 1.8rem;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 0.8rem;
  transition: all 0.3s ease;
  border: none;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  box-shadow: 0 4px 10px rgba(255, 133, 0, 0.3);
  
  &:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(255, 133, 0, 0.4);
  }
  
  &:active {
    transform: translateY(-1px);
  }
  
  .icon {
    font-size: 1.3rem;
  }
  
  &.secondary {
    background: linear-gradient(135deg, ${colors.surfaceLight} 0%, ${colors.surface} 100%);
    color: ${colors.accent};
    border: 2px solid ${colors.accent};
    box-shadow: none;
    
    &:hover {
      background: rgba(255, 133, 0, 0.1);
    }
  }
`;

export const TeamSection = styled.div`
  margin-top: 4rem;
  text-align: center;
  
  h2 {
    color: ${colors.accent};
    margin: 0 0 2.5rem 0;
    font-size: 2.2rem;
    position: relative;
    display: inline-block;
    padding-bottom: 0.8rem;
    
    &:after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 20%;
      right: 20%;
      height: 3px;
      background: linear-gradient(to right, transparent, ${colors.accent}, transparent);
    }
  }
`;

export const TeamGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
`;

export const TeamMemberCard = styled.div`
  background: linear-gradient(135deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
  }
  
  .photo {
    width: 100%;
    height: 250px;
    overflow: hidden;
    position: relative;
    
    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: all 0.5s ease;
    }
    
    &:after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 50%;
      background: linear-gradient(to top, ${colors.surfaceDark}, transparent);
    }
  }
  
  &:hover .photo img {
    transform: scale(1.05);
  }
  
  .info {
    padding: 1.5rem;
    text-align: center;
    
    h3 {
      margin: 0 0 0.5rem 0;
      color: ${colors.accent};
      font-size: 1.3rem;
    }
    
    .role {
      color: ${colors.textSecondary};
      font-size: 0.9rem;
      margin-bottom: 1rem;
      display: block;
    }
    
    p {
      color: ${colors.textMuted};
      font-size: 0.95rem;
      line-height: 1.5;
      margin-bottom: 1rem;
    }
  }
  
  .social {
    display: flex;
    justify-content: center;
    gap: 1rem;
    padding-bottom: 0.5rem;
    
    a {
      color: ${colors.textSecondary};
      transition: all 0.3s ease;
      
      &:hover {
        color: ${colors.accent};
      }
      
      .icon {
        font-size: 1.2rem;
      }
    }
  }
`;

export const Timeline = styled.div`
  margin: 3rem 0;
  position: relative;
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 3px;
    background: linear-gradient(to bottom, transparent, ${colors.accent}, transparent);
    transform: translateX(-50%);
  }
`;

export const TimelineItem = styled.div<{ align?: 'left' | 'right' }>`
  display: flex;
  justify-content: ${props => props.align === 'right' ? 'flex-end' : 'flex-start'};
  padding-bottom: 3rem;
  position: relative;
  width: 50%;
  align-self: ${props => props.align === 'right' ? 'flex-end' : 'flex-start'};
  margin-left: ${props => props.align === 'right' ? '50%' : '0'};
  
  .content {
    background: linear-gradient(135deg, ${colors.surface} 0%, ${colors.surfaceDark} 100%);
    border-radius: 10px;
    padding: 1.5rem;
    width: calc(100% - 30px);
    position: relative;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    border-left: 4px solid ${colors.accent};
    
    h3 {
      margin: 0 0 0.5rem 0;
      color: ${colors.accent};
      font-size: 1.3rem;
    }
    
    .date {
      color: ${colors.textMuted};
      font-size: 0.9rem;
      margin-bottom: 1rem;
      display: block;
    }
    
    p {
      color: ${colors.textSecondary};
      font-size: 1rem;
      line-height: 1.6;
      margin: 0;
    }
    
    &:before {
      content: '';
      position: absolute;
      top: 20px;
      ${props => props.align === 'right' ? 'left: -10px; transform: rotate(45deg);' : 'right: -10px; transform: rotate(-135deg);'}
      width: 20px;
      height: 20px;
      background: ${colors.surface};
      border-top: 1px solid ${colors.border};
      border-right: 1px solid ${colors.border};
    }
  }
  
  .dot {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: ${colors.accent};
    position: absolute;
    top: 20px;
    left: ${props => props.align === 'right' ? '-10px' : 'calc(100% - 10px)'};
    z-index: 2;
    box-shadow: 0 0 0 4px rgba(255, 133, 0, 0.3);
  }
`;
