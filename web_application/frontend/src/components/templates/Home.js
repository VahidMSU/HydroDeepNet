import React, { useState } from 'react';
import { 
  Box, Typography, List, ListItem, ListItemText, Grid, Paper, Button, Dialog, DialogTitle, DialogContent, IconButton 
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const HomeTemplate = () => {
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState('');

  const handleOpenModal = (imageSrc) => {
    setModalImage(imageSrc);
    setModalOpen(true);
  };

  const handleCloseModal = () => {
    setModalOpen(false);
  };

  return (
    <Box sx={{ bgcolor: '#2b2b2c', minHeight: '100vh', px: 4, py: 3 }}>
      <Typography variant="h3" align="center" gutterBottom sx={{ fontWeight: 'bold', color: 'white' }}>
        Hydrological Modeling and Deep Learning Framework
      </Typography>

      {/* Overview Section */}
      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <Typography variant="h4" sx={{ color: '#ff8500', mb: 2, fontWeight: 'bold' }}>Overview</Typography>
        <Typography sx={{color: 'white'}}>
          This platform integrates advanced hydrological modeling, hierarchical data management, and deep learning techniques.
          It leverages models such as SWAT+ and MODFLOW to predict hydrological variables at high spatial and temporal resolutions.
        </Typography>
      </Paper>

      {/* Key Components Section */}
      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <Typography variant="h4" sx={{ color: '#ff8500', mb: 2, fontWeight: 'bold' }}>Key Components</Typography>

        {/* Hydrological Modeling */}
        <Typography variant="h5" sx={{ mt: 2, fontWeight: 'bold', color: 'white' }}>1. Hydrological Modeling with SWAT+</Typography>
        <Typography sx={{color: 'white', fontStyle: 'italic', fontWeight: 'bold' }}>
          SWAT+ serves as the core model for simulating surface and subsurface hydrological cycles. Key highlights:
        </Typography>
        <List sx={{ pl: 3, color: 'white', listStyleType: 'disc' }}>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Simulates evapotranspiration, runoff, and groundwater recharge." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Uses hierarchical land classification for HRU-based analysis." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Employs Particle Swarm Optimization (PSO) for calibrating parameters." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
        </List>

        {/* Hierarchical Data Management */}
        <Typography variant="h5" sx={{ mt: 2, fontWeight: 'bold', color: 'white' }}>2. Hierarchical Data Management</Typography>
        <Typography sx={{color: 'white', fontStyle: 'italic', fontWeight: 'bold' }}>The platform uses a robust HDF5 database to manage multi-resolution data.</Typography>
        <List sx={{ pl: 3, color: 'white', listStyleType: 'disc' }}>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Land use and soil data (250m resolution)." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Groundwater hydraulic properties from 650k water wells." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Meteorological inputs from PRISM (4km) and NSRDB (2km, upsampled to 4km)." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
        </List>

        {/* GeoNet Vision System */}
        <Typography variant="h5" sx={{ mt: 2, fontWeight: 'bold', color: 'white' }}>3. GeoNet Vision System</Typography>
        <Typography sx={{color: 'white', fontStyle: 'italic', fontWeight: 'bold' }}>
          GeoNet leverages hydrological data for spatiotemporal regression tasks, predicting groundwater recharge and climate impacts.
        </Typography>
        <List sx={{ pl: 3, color: 'white', listStyleType: 'disc' }}>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Support for 4D spatiotemporal analysis at 250m resolution." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Efficient processing of hydrological data with specialized loss functions." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Modular design for hyperparameter tuning and model customization." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
        </List>
      </Paper>

      {/* Image Section */}
      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <Typography variant="h4" sx={{ color: '#ff8500', mb: 2, fontWeight: 'bold' }}>
          Hydrological Model Creation Framework
        </Typography>

        {/* Centering the image and making it larger */}
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mt: 2 }}>
          <Paper sx={{ bgcolor: '#2b2b2c', p: 2, borderRadius: 2 }}>
            <img
              src="/static/images/SWATGenX_flowchart.jpg"
              alt="SWATGenX Workflow"
              style={{ width: '200%', maxWidth: '800px', borderRadius: 10, cursor: 'pointer' }}
              onClick={() => handleOpenModal('/static/images/SWATGenX_flowchart.jpg')}
            />
            <Typography variant="body1" sx={{ mt: 1, color: 'white', textAlign: 'center' }}>
              SWATGenX Workflow
            </Typography>
          </Paper>
        </Box>
      </Paper>

      {/* Applications Section */}
      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <Typography variant="h4" sx={{ color: '#ff8500', mb: 2, fontWeight: 'bold' }}>Applications</Typography>
        <List sx={{ pl: 3, color: 'white', listStyleType: 'disc' }}>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Predicting groundwater recharge in data-scarce regions." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Assessing climate change impacts on hydrological processes." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText 
              primary="Supporting scalable watershed-level hydrological modeling." 
              sx={{ color: 'white' }} 
              slotProps={{ primary: { sx: { fontWeight: 'medium' } } }}
            />
          </ListItem>
        </List>
      </Paper>

      {/* Buttons Section */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 3 }}>
        <Button variant="contained" color="primary" sx={{ bgcolor: '#ff8500' }} onClick={() => alert('Learn More clicked!')}>
          Learn More
        </Button>
        <Button variant="contained" color="primary" sx={{ bgcolor: '#ff8500' }} onClick={() => alert('Download Data clicked!')}>
          Download Data
        </Button>
      </Box>

      {/* Image Modal */}
      <Dialog open={modalOpen} onClose={handleCloseModal} maxWidth="md" fullWidth>
        <DialogTitle>
          <IconButton edge="end" color="inherit" onClick={handleCloseModal} aria-label="close">
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <img src={modalImage} alt="Enlarged View" style={{ width: '100%', borderRadius: 5 }} />
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default HomeTemplate;




// Original Backup

// import React from 'react';
// import {
//   Section,
//   Header,
//   SubHeader,
//   Paragraph,
//   List,
//   ImageGrid,
//   ImageCard,
//   InteractiveButtons,
//   Modal,
//   ModalClose,
//   HeaderTitle,
// } from '../../styles/Layout.tsx';
// import { Container } from '../../styles/Home.tsx';

// const HomeTemplate = () => {
//   return (
//     <div>
//       <Container>
//         <HeaderTitle>Hydrological Modeling and Deep Learning Framework</HeaderTitle>

//         <Section>
//           <Header>Overview</Header>
//           <Paragraph>
//             This platform integrates advanced hydrological modeling, hierarchical data management,
//             and deep learning techniques. By leveraging models such as SWAT+ and MODFLOW, it
//             predicts hydrological variables at high spatial and temporal resolutions, enabling
//             improved understanding of surface and groundwater processes.
//           </Paragraph>
//         </Section>

//         <Section>
//           <Header>Key Components</Header>

//           <SubHeader>1. Hydrological Modeling with SWAT+</SubHeader>
//           <Paragraph>
//             SWAT+ serves as the core model for simulating surface and subsurface hydrological
//             cycles. It integrates datasets such as NHDPlus HR (1:24k resolution) and DEM (30m
//             resolution) to accurately model watershed processes. Key highlights:
//           </Paragraph>
//           <List>
//             <li>Simulates evapotranspiration, runoff, and groundwater recharge.</li>
//             <li>Uses hierarchical land classification for HRU-based analysis.</li>
//             <li>
//               Employs Particle Swarm Optimization (PSO) for calibrating hydrological parameters.
//             </li>
//           </List>

//           <SubHeader>2. Hierarchical Data Management</SubHeader>
//           <Paragraph>
//             The platform uses a robust HDF5 database to manage multi-resolution data, integrating
//             datasets like:
//           </Paragraph>
//           <List>
//             <li>Land use and soil data (250m resolution).</li>
//             <li>
//               Groundwater hydraulic properties derived by Empirical Bayesian Kriging (EBK) from 650k
//               water wells observation.
//             </li>
//             <li>Meteorological inputs from PRISM (4km) and NSRDB (2km, upsampled to 4km).</li>
//           </List>
//           <Paragraph>
//             Temporal and spatial preprocessing ensures consistent resolution and gap-filling for
//             missing data.
//           </Paragraph>

//           <SubHeader>3. GeoNet Vision System</SubHeader>
//           <Paragraph>
//             GeoNet leverages hydrological data to perform spatiotemporal regression tasks. Using
//             CNN-Transformers and other deep learning architectures, GeoNet predicts groundwater
//             recharge, climate impacts, and more. Features include:
//           </Paragraph>
//           <List>
//             <li>Support for 4D spatiotemporal analysis at 250m resolution.</li>
//             <li>
//               Efficient processing of hydrological data with specialized loss functions for spatial
//               and temporal evaluation.
//             </li>
//             <li>Modular design for hyperparameter tuning and model customization.</li>
//           </List>
//         </Section>

//         <Section>
//           <Header>Hydrological Model Creation Framework</Header>
//           <ImageGrid>
//             <ImageCard>
//               <img
//                 src={`/static/images/SWATGenX_flowchart.jpg`}
//                 alt="SWATGenX Workflow"
//                 onClick={() => openModal(`/static/images/SWATGenX_flowchart.jpg`)}
//               />
//               <h4>SWATGenX Workflow</h4>
//             </ImageCard>
//           </ImageGrid>
//         </Section>

//         <Section>
//           <Header>Applications</Header>
//           <List>
//             <li>Predicting groundwater recharge in data-scarce regions.</li>
//             <li>Assessing the impacts of climate change on hydrological processes.</li>
//             <li>Supporting scalable watershed-level hydrological modeling and decision-making.</li>
//           </List>
//         </Section>

//         <InteractiveButtons>
//           <button
//             onClick={() => alert('Learn More clicked!')}
//             style={{ textDecoration: 'none', color: 'inherit' }}
//           >
//             Learn More
//           </button>
//           <button
//             onClick={() => alert('Download Data clicked!')}
//             style={{ textDecoration: 'none', color: 'inherit' }}
//           >
//             Download Data
//           </button>
//         </InteractiveButtons>

//         <Modal id="imageModal">
//           <ModalClose onClick={closeModal}>&times;</ModalClose>
//           <img id="modalImage" src="" alt="Large view" />
//         </Modal>
//       </Container>
//     </div>
//   );
// };

// // Open the modal with the clicked image
// const openModal = (imageSrc) => {
//   const modal = document.getElementById('imageModal');
//   const modalImage = document.getElementById('modalImage');

//   modal.style.display = 'block';
//   modalImage.src = imageSrc;
// };

// // Close the modal
// const closeModal = () => {
//   const modal = document.getElementById('imageModal');
//   modal.style.display = 'none';
// };

// export default HomeTemplate;
