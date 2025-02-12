import React, { useState } from 'react';
import { 
  Box, Typography, List, ListItem, ListItemText, Paper, Button, Dialog, DialogTitle, DialogContent, IconButton 
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
