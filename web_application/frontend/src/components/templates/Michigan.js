import React, { useState } from 'react';
import { Box, Typography, Grid, Card, CardMedia, Dialog, IconButton } from '@mui/material';
import { styled } from '@mui/system';
import CloseIcon from '@mui/icons-material/Close';

// Styled Components
const MichiganContainer = styled(Box)({
  marginTop: '2rem',
  marginLeft: '2rem',
  marginRight: '2rem',
  padding: '4rem',
  borderRadius: '16px',
  boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)',
  textAlign: 'center',
});

const MichiganTitle = styled(Typography)({
  color: 'white',
  fontSize: '2.5rem',
  textAlign: 'center',
  marginBottom: '1rem',
  fontWeight: 'bold',
});

const CardTitle = styled(Typography)({
  color: '#ffffff',
  fontSize: '1rem',
  textAlign: 'center',
  padding: '0.5rem',
  fontWeight: 600,
  backgroundColor: '#222',
  borderBottom: '4px solid #ff5722', // Orange bottom border
  borderBottomLeftRadius: '8px',
  borderBottomRightRadius: '8px',
});

const ModalImage = styled(CardMedia)({
  maxWidth: '90%',
  maxHeight: '80vh',
  borderRadius: '8px',
});

// Component
const MichiganTemplate = () => {
  const [modalOpen, setModalOpen] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);

  const images = [
    {
      src: `/static/images/models_boundary_huc12_huc8.jpeg`,
      alt: 'Model Boundaries for HUC12 and HUC8 Models',
      title: 'Boundaries of the Established Hydrological Models',
    },
    {
      src: `/static/images/simple_swat_example.jpeg`,
      alt: 'Simple SWAT+ Model Example',
      title: 'Examples of SWAT+ Models',
    },
    {
      src: `/static/images/complex_swat_example.jpeg`,
      alt: 'Complex SWAT+ Model Example',
      title: 'Complex SWAT+ Model Example',
    },
    {
      src: `/static/images/SWATplus_hrus_area.jpeg`,
      alt: 'SWATplus HRUs Area',
      title: 'Distribution of Hydrologic Response Units (HRUs)',
    },
    {
      src: `/static/images/Daily_all_stations_with_distribution.png`,
      alt: 'Daily SWAT gwflow MODEL Performance',
      title: 'Daily SWAT+ Models Performance',
    },
    {
      src: `/static/images/Monthly_all_stations_with_distribution.png`,
      alt: 'Monthly SWAT MODEL Performance',
      title: 'Monthly SWAT+ Models Performance',
    },
    {
      src: `/static/images/MODFLOW_model_input_example.jpeg`,
      alt: 'MODFLOW Model Example',
      title: 'MODFLOW Model Example',
    },
    {
      src: `/static/images/EBK_metrics.jpeg`,
      alt: 'MODFLOW Model Performances',
      title: 'Distribution of MODFLOW Model Performances',
    },
    {
      src: `/static/images/average_hours_vs_HRU_n_rivers.png`,
      alt: 'Simulation Time vs HRU and Rivers',
      title: 'Simulation Time vs HRU and Rivers',
    },
  ];

  const openModal = (image) => {
    setCurrentImage(image);
    setModalOpen(true);
  };

  const closeModal = (e) => {
    if (e.target.tagName !== 'IMG') {
      setModalOpen(false);
      setCurrentImage(null);
    }
  };

  return (
    <MichiganContainer>
      <MichiganTitle variant="h1">Michigan LP Hydrologic Modeling Coverage and Performance</MichiganTitle>

      {/* Grid Layout with Two Columns */}
      <Grid container spacing={2} justifyContent="center" alignItems="center">
        {images.map((image, index) => (
          <Grid 
            item 
            xs={12} 
            sm={6} 
            key={index} 
            sx={{ display: 'flex', justifyContent: 'center', flexGrow: 1 }} // Ensures last image is centered properly
          >
            <Card sx={{ cursor: 'pointer', borderRadius: '8px', backgroundColor: '#222', width: '100%', maxWidth: '500px', mt: 3 }} onClick={() => openModal(image)}>
              <CardMedia component="img" image={image.src} alt={image.alt} sx={{ width: '100%', borderRadius: '8px 8px 0 0' }} />
              <CardTitle>{image.title}</CardTitle>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Modal for Viewing Images */}
      <Dialog open={modalOpen} onClose={closeModal} maxWidth="md" fullWidth>
        <Box position="relative" bgcolor="black" display="flex" justifyContent="center" alignItems="center" height="100vh">
          <IconButton sx={{ position: 'absolute', top: 10, right: 10, color: 'white' }} onClick={closeModal}>
            <CloseIcon />
          </IconButton>
          {currentImage && <ModalImage component="img" image={currentImage.src} alt={currentImage.alt} />}
        </Box>
      </Dialog>
    </MichiganContainer>
  );
};

export default MichiganTemplate;