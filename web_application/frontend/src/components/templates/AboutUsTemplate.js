import React from 'react';
import {
  AboutContainer,
  SectionTitle,
  ContentSection,
  ContactSection,
  ProjectLinks,
} from '../../styles/AboutUsTemplate.tsx';

const AboutUsTemplate = () => {
  return (
    <AboutContainer>
      <SectionTitle>About HydroDeepNet</SectionTitle>

      <ContentSection>
        <h3>Project Overview</h3>
        <p>
          HydroDeepNet is an innovative platform developed and designed to streamline the creation and advancement of hydrological and
          deep learning models. This project represents a significant advancement in predictive
          ability for estimating various hydrological variables required for accurate water
          management, making it more accessible and efficient for researchers and practitioners.
        </p>

        <h3>Mission</h3>
        <p>
          Our mission is to democratize hydrological modeling by providing researchers and water
          resource managers with powerful, user-friendly tools for creating and analyzing SWAT+
          models. We aim to facilitate better understanding and management of water resources
          through advanced modeling capabilities.
        </p>

        <h3>Key Features</h3>
        <ul>
          <li>Automated SWAT+ model generation for any USGS streamgage station</li>
          <li>Integrated calibration and validation capabilities</li>
          <li>Advanced sensitivity analysis tools</li>
          <li>User-friendly interface for model parameterization</li>
        </ul>
      </ContentSection>

      <ContactSection>
        <h3>Contact & Resources</h3>
        <ProjectLinks>
          <a href="mailto:rafieiva@msu.edu">Contact Us</a>
          <a
            href="https://bitbucket.org/vahidrafiei/swatgenx/src/main/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Source Code Repository
          </a>
        </ProjectLinks>
      </ContactSection>
    </AboutContainer>
  );
};

export default AboutUsTemplate;
