import React from 'react';
import {
  Section,
  Header,
  SubHeader,
  Paragraph,
  List,
  ImageGrid,
  ImageCard,
  InteractiveButtons,
  Modal,
  ModalClose,
  HeaderTitle,
} from '../../styles/Layout.tsx';
import { Body, Container } from '../../styles/Home.tsx';

const HomeTemplate = () => {
  return (
    <Body>
      <Container>
        <HeaderTitle>Hydrological Modeling and Deep Learning Framework</HeaderTitle>

        <Section>
          <Header>Overview</Header>
          <Paragraph>
            This platform integrates advanced hydrological modeling, hierarchical data management,
            and deep learning techniques. By leveraging models such as SWAT+ and MODFLOW, it
            predicts hydrological variables at high spatial and temporal resolutions, enabling
            improved understanding of surface and groundwater processes.
          </Paragraph>
        </Section>

        <Section>
          <Header>Key Components</Header>

          <SubHeader>1. Hydrological Modeling with SWAT+</SubHeader>
          <Paragraph>
            SWAT+ serves as the core model for simulating surface and subsurface hydrological
            cycles. It integrates datasets such as NHDPlus HR (1:24k resolution) and DEM (30m
            resolution) to accurately model watershed processes. Key highlights:
          </Paragraph>
          <List>
            <li>Simulates evapotranspiration, runoff, and groundwater recharge.</li>
            <li>Uses hierarchical land classification for HRU-based analysis.</li>
            <li>
              Employs Particle Swarm Optimization (PSO) for calibrating hydrological parameters.
            </li>
          </List>

          <SubHeader>2. Hierarchical Data Management</SubHeader>
          <Paragraph>
            The platform uses a robust HDF5 database to manage multi-resolution data, integrating
            datasets like:
          </Paragraph>
          <List>
            <li>Land use and soil data (250m resolution).</li>
            <li>
              Groundwater hydraulic properties derived by Empirical Bayesian Kriging (EBK) from 650k
              water wells observation.
            </li>
            <li>Meteorological inputs from PRISM (4km) and NSRDB (2km, upsampled to 4km).</li>
          </List>
          <Paragraph>
            Temporal and spatial preprocessing ensures consistent resolution and gap-filling for
            missing data.
          </Paragraph>

          <SubHeader>3. GeoNet Vision System</SubHeader>
          <Paragraph>
            GeoNet leverages hydrological data to perform spatiotemporal regression tasks. Using
            CNN-Transformers and other deep learning architectures, GeoNet predicts groundwater
            recharge, climate impacts, and more. Features include:
          </Paragraph>
          <List>
            <li>Support for 4D spatiotemporal analysis at 250m resolution.</li>
            <li>
              Efficient processing of hydrological data with specialized loss functions for spatial
              and temporal evaluation.
            </li>
            <li>Modular design for hyperparameter tuning and model customization.</li>
          </List>
        </Section>

        <Section>
          <Header>Hydrological Model Creation Framework</Header>
          <ImageGrid>
            <ImageCard>
              <img
                src={`/static/images/SWATGenX_flowchart.jpg`}
                alt="SWATGenX Workflow"
                onClick={() => openModal(`/static/images/SWATGenX_flowchart.jpg`)}
              />
              <h4>SWATGenX Workflow</h4>
            </ImageCard>
          </ImageGrid>
        </Section>

        <Section>
          <Header>Applications</Header>
          <List>
            <li>Predicting groundwater recharge in data-scarce regions.</li>
            <li>Assessing the impacts of climate change on hydrological processes.</li>
            <li>Supporting scalable watershed-level hydrological modeling and decision-making.</li>
          </List>
        </Section>

        <InteractiveButtons>
          <button
            onClick={() => alert('Learn More clicked!')}
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            Learn More
          </button>
          <button
            onClick={() => alert('Download Data clicked!')}
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            Download Data
          </button>
        </InteractiveButtons>

        <Modal id="imageModal">
          <ModalClose onClick={closeModal}>&times;</ModalClose>
          <img id="modalImage" src="" alt="Large view" />
        </Modal>
      </Container>
    </Body>
  );
};

// Open the modal with the clicked image
const openModal = (imageSrc) => {
  const modal = document.getElementById('imageModal');
  const modalImage = document.getElementById('modalImage');

  modal.style.display = 'block';
  modalImage.src = imageSrc;
};

// Close the modal
const closeModal = () => {
  const modal = document.getElementById('imageModal');
  modal.style.display = 'none';
};

export default HomeTemplate;
