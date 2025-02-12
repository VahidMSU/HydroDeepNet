import React from 'react';
import {
  AboutContainer,
  SectionTitle,
  ContentSection,
  ContactSection,
  ProjectLinks,
} from '../../styles/AboutUs.tsx';

const AboutUsTemplate = () => {
  return (
    <AboutContainer>
      <SectionTitle>About HydroDeepNet</SectionTitle>

      <ContentSection>
        <h3>Mission</h3>
        <p>
          HydroDeepNet is a web-based platform that provides hydrological modeling tools for
          researchers, practitioners, and students. Our mission is to democratize hydrological
          modeling by offering user-friendly interfaces, automation, and deep learning capabilities
          for hydrological research and applications.
        </p>
        <h3>SWATGenX</h3>
        <p>
          SWATGenX automates the creation of SWAT+ models for any USGS streamgage station. Users can
          search by station name or site number, configure calibration, sensitivity analysis, and
          validation settings, and generate hydrological models directly from the platform.
        </p>

        <h3>Vision System for Deep Learning</h3>
        <p>
          The Vision System enables deep learning-based hydrological modeling. It allows users to
          design, train, and deploy models with custom configurations to predict hydrological
          variables using satellite, climate, and geospatial datasets.
        </p>

        <h3>HydroGeoDataset</h3>
        <p>
          HydroGeoDataset compiles national datasets such as MODIS, PRISM, LOCA2, and Wellogic,
          alongside deep learning-derived data. It provides high-resolution hydrological,
          geological, and climate data for Michigan's Lower Peninsula.
        </p>

        <h3>Modeling & Data Outputs</h3>
        <p>
          Users can generate spatiotemporal simulations, time-series predictions, and
          high-resolution GIS outputs. The system supports real-time data visualization, model
          comparisons, and downloadable results for further analysis.
        </p>
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
