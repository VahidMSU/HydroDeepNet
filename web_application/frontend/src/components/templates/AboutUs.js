import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faChevronDown,
  faRobot,
  faDatabase,
  faChartLine,
  faMap,
  faServer,
  faCode,
  faBook,
  faFileAlt,
  faGlobe,
  faUniversity,
} from '@fortawesome/free-solid-svg-icons';

import {
  AboutUsContainer,
  AboutUsHeader,
  AccordionContainer,
  AccordionItem,
  AccordionHeader,
  AccordionContent,
  FeatureCard,
  ContactSection,
  ButtonContainer,
  AboutUsButton,
  Timeline,
  TimelineItem,
} from '../../styles/AboutUs.tsx';

const AboutUsTemplate = () => {
  // State to track which accordion is open
  const [openAccordion, setOpenAccordion] = useState(0);

  // Function to handle accordion toggle
  const toggleAccordion = (index) => {
    setOpenAccordion(openAccordion === index ? -1 : index);
  };

  // Accordion content data
  const accordionData = [
    {
      title: 'Mission',
      icon: faGlobe,
      content: `HydroDeepNet is a web-based platform that provides hydrological modeling tools for
      researchers, practitioners, and students. Our mission is to democratize hydrological
      modeling by offering user-friendly interfaces, automation, and deep learning
      capabilities for hydrological research and applications.`,
    },
    {
      title: 'SWATGenX',
      icon: faMap,
      content: `SWATGenX automates the creation of SWAT+ models for any USGS streamgage station. Users
      can search by station name or site number, configure landuse/soil/DEM resolutions, opt
      for calibration, sensitivity analysis, and validation settings, and generate
      hydrological models directly from the platform.`,
    },
    {
      title: 'Vision System for Deep Learning',
      icon: faRobot,
      content: `The Vision System enables deep learning-based hydrological modeling. It allows users to
      design, train, and deploy models with custom configurations to predict hydrological
      variables using satellite, climate, and geospatial datasets.`,
    },
    {
      title: 'HydroGeoDataset',
      icon: faDatabase,
      content: `HydroGeoDataset compiles national datasets such as MODIS, PRISM, LOCA2, and Wellogic,
      alongside deep modeling and deep learning-derived data. It provides high-resolution
      hydrological, geological, and climate data for Michigan's Lower Peninsula.`,
    },
    {
      title: 'Modeling & Data Outputs',
      icon: faChartLine,
      content: `Users can generate spatiotemporal simulations, time-series predictions, and
      high-resolution GIS outputs in HDS format. The system supports real-time data
      visualization, model comparisons, and downloadable results for further analysis.`,
    },
  ];

  // Key features data
  const featuresData = [
    {
      icon: faServer,
      title: 'Automated Model Creation',
      description:
        'Build complex hydrological models in minutes instead of days with our automated model generation system, which handles everything from data acquisition to parameter setting.',
    },
    {
      icon: faRobot,
      title: 'AI-Powered Predictions',
      description:
        'Leverage state-of-the-art deep learning algorithms specifically designed for hydrological modeling to make accurate predictions about water resource behavior.',
    },
    {
      icon: faDatabase,
      title: 'Comprehensive Dataset Access',
      description:
        'Access a vast collection of hydrological, geological, and climate data all in one place, with user-friendly query tools and visualization capabilities.',
    },
  ];

  // Timeline data
  const timelineData = [
    {
      year: '2020',
      title: 'Project Inception',
      description:
        'HydroDeepNet was conceptualized as a comprehensive platform for hydrological modeling and analysis.',
    },
    {
      year: '2021',
      title: 'SWATGenX Development',
      description:
        'Development of the automated SWAT+ model generation tool, making hydrological modeling more accessible.',
    },
    {
      year: '2022',
      title: 'Vision System Launch',
      description:
        'Introduction of deep learning capabilities with the Vision System for advanced hydrological predictions.',
    },
    {
      year: '2023',
      title: 'HydroGeoDataset Integration',
      description:
        'Compilation and integration of comprehensive hydrological and geological datasets for Michigan.',
    },
  ];

  return (
    <AboutUsContainer>
      <AboutUsHeader>
        <h1>About HydroDeepNet</h1>
        <p>
          Advancing hydrological science through automation, deep learning, and accessible tools for
          researchers, practitioners, and students worldwide.
        </p>
      </AboutUsHeader>

      {/* Accordion Sections */}
      <AccordionContainer>
        {accordionData.map((item, index) => (
          <AccordionItem key={index}>
            <AccordionHeader
              isOpen={openAccordion === index}
              onClick={() => toggleAccordion(index)}
            >
              <h3>
                <FontAwesomeIcon icon={item.icon} className="icon-title" />
                {item.title}
              </h3>
              <FontAwesomeIcon icon={faChevronDown} className="icon" />
            </AccordionHeader>
            <AccordionContent isOpen={openAccordion === index}>
              <p>{item.content}</p>
            </AccordionContent>
          </AccordionItem>
        ))}
      </AccordionContainer>

      {/* Key Features */}
      <section>
        <h2
          style={{ color: '#ff5722', textAlign: 'center', marginBottom: '2rem', fontSize: '2rem' }}
        >
          Key Features
        </h2>

        {featuresData.map((feature, index) => (
          <FeatureCard key={index}>
            <div className="icon-container">
              <FontAwesomeIcon icon={feature.icon} className="icon" />
            </div>
            <div className="content">
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          </FeatureCard>
        ))}
      </section>

      {/* Project Timeline */}
      <section>
        <h2
          style={{
            color: '#ff5722',
            textAlign: 'center',
            marginTop: '4rem',
            marginBottom: '2rem',
            fontSize: '2rem',
          }}
        >
          Project Timeline
        </h2>

        <Timeline>
          {timelineData.map((item, index) => (
            <TimelineItem key={index} align={index % 2 === 0 ? 'left' : 'right'}>
              <div className="dot"></div>
              <div className="content">
                <h3>{item.title}</h3>
                <span className="date">{item.year}</span>
                <p>{item.description}</p>
              </div>
            </TimelineItem>
          ))}
        </Timeline>
      </section>

      {/* Contact & Resources */}
      <ContactSection>
        <h2>Resources</h2>
        <p>
          Explore our documentation and source code repositories for more information about
          HydroDeepNet.
        </p>
        <ButtonContainer>
          <AboutUsButton
            href="https://bitbucket.org/hydrodeepnet/swatgenx/src/main/"
            target="_blank"
          >
            <FontAwesomeIcon icon={faCode} className="icon" />
            Source Code
          </AboutUsButton>
          <AboutUsButton href="#documentation" className="secondary">
            <FontAwesomeIcon icon={faBook} className="icon" />
            Documentation
          </AboutUsButton>
          <AboutUsButton href="#publications" className="secondary">
            <FontAwesomeIcon icon={faFileAlt} className="icon" />
            Publications
          </AboutUsButton>
          <AboutUsButton href="https://msu.edu" target="_blank" className="secondary">
            <FontAwesomeIcon icon={faUniversity} className="icon" />
            Michigan State University
          </AboutUsButton>
        </ButtonContainer>
      </ContactSection>
    </AboutUsContainer>
  );
};

export default AboutUsTemplate;
