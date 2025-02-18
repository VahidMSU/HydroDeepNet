import React from 'react';
import {
  TermsContainer,
  TermsTitle,
  SectionHeader,
  TermsText,
  ContentWrapper,
  ContactSection,
  Links,
} from '../../styles/TermsConditions.tsx';

const TermsAndConditionsTemplate = () => {
  return (
    <TermsContainer>
      <TermsTitle>Terms and Conditions</TermsTitle>

      <ContentWrapper>
        <SectionHeader>Overview</SectionHeader>
        <TermsText>
          This web application is provided as a research collaboration tool for hydrological and
          environmental modeling. By accessing and using this platform, you agree to the following
          terms.
        </TermsText>

        <SectionHeader>Data Sources and Technology</SectionHeader>
        <TermsText>
          The application is built on open-source technologies and integrates datasets from public
          sources, including the National Solar Radiation Database (NSRDB), NHDPlus High Resolution,
          LANDFIRE, STATSGO2, and SNODAS. The platform also utilizes software such as QSWAT+,
          SWATPlusEditor, SWAT+, MODFLOW, and FloPy for modeling workflows.
        </TermsText>

        <SectionHeader>User Responsibilities</SectionHeader>
        <TermsText>
          Users are responsible for ensuring that their activities comply with Michigan State
          University's IT policies and relevant research agreements. Unauthorized access, data
          scraping, or any form of misuse, including attempts to bypass authentication or disrupt
          system functionality, is strictly prohibited.
        </TermsText>

        <SectionHeader>Data Usage</SectionHeader>
        <TermsText>
          Access to specific datasets may be subject to licensing agreements or institutional
          policies. Users contributing data must ensure they have the appropriate permissions to
          share it on this platform.
        </TermsText>

        <ContactSection>
          <SectionHeader>Contact Information</SectionHeader>
          <Links>
            <a href="mailto:rafieiva@msu.edu">Contact Us</a>
            <a href="https://msu.edu/terms" target="_blank" rel="noopener noreferrer">
              MSU Terms of Use
            </a>
          </Links>
        </ContactSection>
      </ContentWrapper>
    </TermsContainer>
  );
};

export default TermsAndConditionsTemplate;
