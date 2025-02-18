import React from 'react';
import {
  AboutContainer,
  AboutTitle,
  ContentWrapper,
  SectionHeader,
  SectionText,
  ContactSection,
  Links,
} from '../../styles/AboutUs.tsx';

const AboutUsTemplate = () => {
  return (
    <AboutContainer>
      <AboutTitle>About HydroDeepNet</AboutTitle>

      <ContentWrapper>
        <section>
          <SectionHeader>Mission</SectionHeader>
          <SectionText>
            HydroDeepNet is a web-based platform that provides hydrological modeling tools for
            researchers, practitioners, and students. Our mission is to democratize hydrological
            modeling by offering user-friendly interfaces, automation, and deep learning
            capabilities for hydrological research and applications.
          </SectionText>

          <SectionHeader>SWATGenX</SectionHeader>
          <SectionText>
            SWATGenX automates the creation of SWAT+ models for any USGS streamgage station. Users
            can search by station name or site number, configure landuse/soil/DEM resolutions, opt
            for calibration, sensitivity analysis, and validation settings, and generate
            hydrological models directly from the platform.
          </SectionText>

          <SectionHeader>Vision System for Deep Learning</SectionHeader>
          <SectionText>
            The Vision System enables deep learning-based hydrological modeling. It allows users to
            design, train, and deploy models with custom configurations to predict hydrological
            variables using satellite, climate, and geospatial datasets.
          </SectionText>

          <SectionHeader>HydroGeoDataset</SectionHeader>
          <SectionText>
            HydroGeoDataset compiles national datasets such as MODIS, PRISM, LOCA2, and Wellogic,
            alongside deep modeling and deeep learning-derived data. It provides high-resolution
            hydrological, geological, and climate data for Michigan's Lower Peninsula.
          </SectionText>

          <SectionHeader>Modeling & Data Outputs</SectionHeader>
          <SectionText>
            Users can generate spatiotemporal simulations, time-series predictions, and
            high-resolution GIS outputs in HDS format. The system supports real-time data
            visualization, model comparisons, and downloadable results for further analysis.
          </SectionText>

          <ContactSection>
            <SectionHeader>Contact & Resources</SectionHeader>
            <Links>
              <a href="mailto:rafieiva@msu.edu">Contact Us</a>
              <a
                href="https://bitbucket.org/vahidrafiei/swatgenx/src/main/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Source Code Repository
              </a>
            </Links>
          </ContactSection>
        </section>
      </ContentWrapper>
    </AboutContainer>
  );
};

export default AboutUsTemplate;
