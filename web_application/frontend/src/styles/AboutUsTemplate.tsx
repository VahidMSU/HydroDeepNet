import styled from '@emotion/styled';

export const AboutContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  color: #ffffff;
  background-color: #2b2b2c;
`;

export const SectionTitle = styled.h1`
  color: #ff8500;
  font-size: 2.5rem;
  margin-bottom: 2rem;
  text-align: center;
  border-bottom: 3px solid #ff8500;
  padding-bottom: 1rem;
`;

export const ContentSection = styled.section`
  margin-bottom: 3rem;
  background-color: #444e5e;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);

  h3 {
    color: #ff8500;
    margin-bottom: 1rem;
  }

  p {
    line-height: 1.6;
    margin-bottom: 1.5rem;
  }

  ul {
    list-style-type: disc;
    margin-left: 1.5rem;
    margin-bottom: 1.5rem;
  }

  li {
    margin-bottom: 0.5rem;
  }
`;

export const TeamSection = styled.section`
  margin-bottom: 3rem;

  h3 {
    color: #ff8500;
    margin-bottom: 2rem;
  }

  h4 {
    color: #ff8500;
    margin-bottom: 1.5rem;
    font-size: 1.2rem;
  }

  .contributors-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 1.5rem;
  }
`;

export const TeamMember = styled.div`
  display: flex;
  flex-direction: column;
  background-color: #444e5e;
  padding: 1.5rem;
  border-radius: 12px;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);

  .member-info {
    h5 {
      color: #ff8500;
      font-size: 1.1rem;
      margin-bottom: 0.5rem;
    }

    p {
      margin-bottom: 0.3rem;
      font-size: 0.9rem;
    }

    a {
      color: #ff8500;
      text-decoration: none;
      font-size: 0.9rem;

      &:hover {
        text-decoration: underline;
      }
    }
  }

  &.contributor {
    padding: 1rem;
  }
`;

export const ContactSection = styled.section`
  background-color: #444e5e;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);

  h3 {
    color: #ff8500;
    margin-bottom: 1.5rem;
  }
`;

export const ProjectLinks = styled.div`
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;

  a {
    color: #ffffff;
    text-decoration: none;
    background-color: #ff8500;
    padding: 0.8rem 1.5rem;
    border-radius: 6px;
    transition: all 0.2s ease-in-out;

    &:hover {
      background-color: #ffa533;
      transform: translateY(-2px);
    }
  }
`;
