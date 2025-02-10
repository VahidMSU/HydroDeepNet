import React, { useState } from 'react';
import {
  Container,
  HeaderTitle,
  ImageGrid,
  ImageCard,
  Modal,
  ModalClose,
} from '../../styles/MichiganTemplate.tsx'; // Import styled components

const MichiganTemplate = () => {
  const images = [
    {
      src: `${process.env.REACT_APP_PUBLIC_URL}/static/images/models_boundary_huc12_huc8.jpeg`,
      alt: 'Model Boundaries for HUC12 and HUC8 Models',
      title: 'Boundaries of the Established Hydrological Models',
    },
    {
      src: `${process.env.REACT_APP_PUBLIC_URL}/static/images/simple_swat_example.jpeg`,
      alt: 'Simple SWAT+ Model Example',
      title: 'Examples of SWAT+ Models',
    },
    {
      src: `${process.env.REACT_APP_PUBLIC_URL}/static/images/complex_swat_example.jpeg`,
      alt: 'Complex SWAT+ Model Example',
      title: 'Complex SWAT+ Model Example',
    },
    {
      src: `${process.env.REACT_APP_PUBLIC_URL}/static/images/SWATplus_hrus_area.jpeg`,
      alt: 'SWATplus HRUs Area',
      title: 'Distribution of Hydrologic Response Units (HRUs)',
    },
    {
      src: `${process.env.REACT_APP_PUBLIC_URL}/static/images/Daily_all_stations_with_distribution.png`,
      alt: 'Daily SWAT gwflow MODEL Performance',
      title: 'Daily SWAT+ Models Performance',
    },
    {
      src: `${process.env.REACT_APP_PUBLIC_URL}/static/images/Monthly_all_stations_with_distribution.png`,
      alt: 'Monthly SWAT MODEL Performance',
      title: 'Monthly SWAT+ Models Performance',
    },
    {
      src: `${process.env.REACT_APP_PUBLIC_URL}/static/images/MODFLOW_model_input_example.jpeg`,
      alt: 'MODFLOW Model Example',
      title: 'MODFLOW Model Example',
    },
    {
      src: `${process.env.REACT_APP_PUBLIC_URL}/static/images/EBK_metrics.jpeg`,
      alt: 'MODFLOW Model Performances',
      title: 'Distribution of MODFLOW Model Performances',
    },
    {
      src: `${process.env.REACT_APP_PUBLIC_URL}/static/images/average_hours_vs_HRU_n_rivers.png`,
      alt: 'Simulation Time vs HRU and Rivers',
      title: 'Simulation Time vs HRU and Rivers',
    },
  ];

  const [modalOpen, setModalOpen] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);

  const openModal = (image) => {
    setCurrentImage(image);
    setModalOpen(true);
  };

  const closeModal = (e) => {
    // Prevent closing if clicking inside the modal image
    if (e.target.tagName !== 'IMG') {
      setModalOpen(false);
      setCurrentImage(null);
    }
  };

  return (
    <Container>
      <HeaderTitle>Michigan LP Hydrologic Modeling Coverage and Performance</HeaderTitle>

      <ImageGrid>
        {images.map((image, index) => (
          <ImageCard key={index}>
            <img
              src={image.src}
              alt={image.alt}
              onClick={() => openModal(image)}
              data-index={index}
            />
            <h4>{image.title}</h4>
          </ImageCard>
        ))}
      </ImageGrid>

      {modalOpen && currentImage && (
        <Modal id="imageModal" onClick={closeModal}>
          <ModalClose onClick={closeModal}>&times;</ModalClose>
          <img
            id="modalImage"
            src={currentImage.src}
            alt={currentImage.alt}
            onClick={(e) => e.stopPropagation()} // Prevent modal close on image click
          />
        </Modal>
      )}
    </Container>
  );
};

export default MichiganTemplate;
