import React from 'react';
import {
  PrivacyContainer,
  PrivacyTitle,
  SectionHeader,
  PolicyText,
  ContentWrapper,
  ContactSection,
  Links,
} from '../../styles/Privacy.tsx';

const PrivacyTemplate = () => {
  return (
    <PrivacyContainer>
      <PrivacyTitle>Privacy Policy</PrivacyTitle>

      <ContentWrapper>
        <SectionHeader>Our Commitment to Privacy</SectionHeader>
        <PolicyText>
          This web application is designed to support research and collaboration by providing access
          to hydrological modeling tools and datasets. We respect user privacy and are committed to
          protecting any information collected during platform use.
        </PolicyText>

        <SectionHeader>Data Collection and Use</SectionHeader>
        <PolicyText>
          We collect user interactions with the application solely for improving system
          functionality, security, and performance. This includes authentication logs and feature
          usage analytics but excludes personally identifiable data beyond what is necessary for
          secure access. All passwords are stored securely using encryption standards.
        </PolicyText>

        <SectionHeader>Data Sharing</SectionHeader>
        <PolicyText>
          User data will not be shared with third parties unless mandated by Michigan State
          University's IT security policies or legal requirements. Any data provided for research
          purposes remains under the ownership of the contributing institution or researcher, and
          its use follows applicable agreements.
        </PolicyText>

        <SectionHeader>Data Sources</SectionHeader>
        <PolicyText>
          The platform operates entirely on open-source software and integrates national datasets,
          including the National Solar Radiation Database (NSRDB), NHDPlus, LANDFIRE, 3D Elevation
          Program, STATSGO2, and SNODAS. We do not collect or process sensitive personal
          information.
        </PolicyText>

        <SectionHeader>Acceptance of Terms</SectionHeader>
        <PolicyText>
          By using this application, you acknowledge that minimal system interaction data is
          collected to improve platform reliability. You may contact the administrators for
          questions regarding data policies or security measures.
        </PolicyText>

        <ContactSection>
          <SectionHeader>Contact & Resources</SectionHeader>
          <Links>
            <a href="mailto:rafieiva@msu.edu">Contact Us</a>
            <a href="https://msu.edu/privacy" target="_blank" rel="noopener noreferrer">
              MSU Privacy Policy
            </a>
          </Links>
        </ContactSection>
      </ContentWrapper>
    </PrivacyContainer>
  );
};

export default PrivacyTemplate;
