import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faMap,
  faChartLine,
  faDiagramProject,
  faWater,
  faCloudSunRain
} from '@fortawesome/free-solid-svg-icons';
import {
  HomeContainer,
  HomeTitle,
  ContentWrapper,
  SectionHeader,
  SectionText,
  ImageContainer,
  CardGrid,
  Card,
  Button,
  ButtonContainer,
  MediaModal,
  MediaWrapper,
  CloseButton
} from '../../styles/Home.tsx';

const HomeTemplate = () => {
  const [selectedMedia, setSelectedMedia] = useState(null);

  const handleMediaClick = (src) => {
    setSelectedMedia(src);
  };

  const handleCloseModal = () => {
    setSelectedMedia(null);
  };

  return (
    <HomeContainer>
      <HomeTitle>Hydrological Modeling and Deep Learning Framework</HomeTitle>

      <ContentWrapper>
        <section>
          <SectionHeader>Overview</SectionHeader>
          <SectionText>
            This platform integrates advanced hydrological modeling, hierarchical data management, and
            deep learning techniques. It leverages models such as SWAT+ and MODFLOW to predict
            hydrological variables at high spatial and temporal resolutions.
          </SectionText>
        </section>

        <section>
          <SectionHeader>Key Components</SectionHeader>
          <SectionText>
            <strong>1. Hydrological Modeling with SWAT+</strong><br />
            SWAT+ serves as the core model for simulating surface and subsurface hydrological cycles.
            Key highlights:
            <ul>
              <li>Simulates evapotranspiration, runoff, and groundwater recharge.</li>
              <li>Uses hierarchical land classification for HRU-based analysis.</li>
              <li>Employs Particle Swarm Optimization (PSO) for calibrating parameters.</li>
            </ul>
          </SectionText>
          
          <SectionText>
            <strong>2. Hierarchical Data Management</strong><br />
            The platform uses a robust HDF5 database to manage multi-resolution data.
            <ul>
              <li>Land use and soil data (250m resolution).</li>
              <li>Groundwater hydraulic properties from 650k water wells.</li>
              <li>Meteorological inputs from PRISM (4km) and NSRDB (2km, upsampled to 4km).</li>
            </ul>
          </SectionText>
          
          <SectionText>
            <strong>3. GeoNet Vision System</strong><br />
            GeoNet leverages hydrological data for spatiotemporal regression tasks, predicting
            groundwater recharge and climate impacts.
            <ul>
              <li>Support for 4D spatiotemporal analysis at 250m resolution.</li>
              <li>Efficient processing of hydrological data with specialized loss functions.</li>
              <li>Modular design for hyperparameter tuning and model customization.</li>
            </ul>
          </SectionText>
        </section>
        
        <section>
          <SectionHeader>HydroDeepNet Workflow</SectionHeader>
          <ImageContainer className="main-image" onClick={() => handleMediaClick('/static/images/SWATGenX_flowchart.jpg')}>
            <img
              src="/static/images/SWATGenX_flowchart.jpg"
              alt="SWATGenX Workflow"
            />
          </ImageContainer>
        </section>
        
        <section>
          <SectionHeader>Applications</SectionHeader>
          <SectionText>
            <ul>
              <li>Predicting groundwater recharge in data-scarce regions.</li>
              <li>Assessing climate change impacts on hydrological processes.</li>
              <li>Supporting scalable watershed-level hydrological modeling.</li>
            </ul>
          </SectionText>
        </section>
        
        <section>
          <SectionHeader>Model Outputs & Visualizations</SectionHeader>
          <SectionText>
            Explore examples of our modeling system outputs and visualizations that showcase the capabilities of SWATGenX.
          </SectionText>
          
          <CardGrid>
            {/* Michigan Card - replaced with generic icon */}
            <Card as={Link} to="/michigan" style={{ textDecoration: 'none' }}>
              <div className="card-image" style={{ 
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #26262a 0%, #1d1d20 100%)'
              }}>
                <FontAwesomeIcon 
                  icon={faMap} 
                  style={{ 
                    fontSize: '5rem', 
                    color: '#ff8500',
                    opacity: 0.8
                  }} 
                />
              </div>
              <div className="card-content">
                <div className="card-title">Michigan LP Modeling</div>
                <div className="card-description">
                  View hydrologic modeling coverage and performance metrics for Michigan's Lower Peninsula watersheds.
                </div>
                <Button as="span">Explore Models</Button>
              </div>
            </Card>
            
            {/* Visualizations Card - replaced with generic icon */}
            <Card as={Link} to="/visualizations" style={{ textDecoration: 'none' }}>
              <div className="card-image" style={{ 
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #26262a 0%, #1d1d20 100%)'
              }}>
                <FontAwesomeIcon 
                  icon={faChartLine} 
                  style={{ 
                    fontSize: '5rem', 
                    color: '#ff8500',
                    opacity: 0.8
                  }} 
                />
              </div>
              <div className="card-content">
                <div className="card-title">SWAT+ Visualizations</div>
                <div className="card-description">
                  Generate custom visualizations from our calibrated SWAT+ models for various watersheds and parameters.
                </div>
                <Button as="span">Generate Visualizations</Button>
              </div>
            </Card>
          </CardGrid>
        </section>
        
        {/* Image Modal */}
        {selectedMedia && (
          <MediaModal onClick={handleCloseModal}>
            <CloseButton onClick={handleCloseModal}>&times;</CloseButton>
            <MediaWrapper onClick={(e) => e.stopPropagation()}>
              <img src={selectedMedia} alt="Enlarged View" />
            </MediaWrapper>
          </MediaModal>
        )}
      </ContentWrapper>
    </HomeContainer>
  );
};

export default HomeTemplate;
