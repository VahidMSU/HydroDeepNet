import React from 'react';
import { Container, Typography, Box, Accordion, AccordionSummary, AccordionDetails, Button } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const AboutUsTemplate = () => {
  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography
        variant="h2"
        fontWeight="bold"
        gutterBottom
        align="center"
        sx={{ color: 'white', mb: 4 }}
      >
        About HydroDeepNet
      </Typography>

      <Box sx={{ mb: 2 }}>
        <Accordion sx={{ bgcolor: 'white', borderRadius: 10, boxShadow: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: 'white' }} />} sx={{ bgcolor: '#444e5e', color: 'white', borderRadius: 1 }}>
            <Typography variant="h5" fontWeight="bold">Mission</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="h6" sx={{ color: "#444e5e" }}>
              HydroDeepNet is a web-based platform that provides hydrological modeling tools for
              researchers, practitioners, and students. Our mission is to democratize hydrological
              modeling by offering user-friendly interfaces, automation, and deep learning
              capabilities for hydrological research and applications.
            </Typography>
          </AccordionDetails>
        </Accordion>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Accordion sx={{ bgcolor: 'white', borderRadius: 10, boxShadow: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: 'white' }} />} sx={{ bgcolor: '#444e5e', color: 'white', borderRadius: 1 }}>
            <Typography variant="h5" fontWeight="bold">SWATGenX</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="h6" sx={{ color: "#444e5e" }}>
              SWATGenX automates the creation of SWAT+ models for any USGS streamgage station. Users
              can search by station name or site number, configure landuse/soil/DEM resolutions, opt
              for calibration, sensitivity analysis, and validation settings, and generate
              hydrological models directly from the platform.
            </Typography>
          </AccordionDetails>
        </Accordion>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Accordion sx={{ bgcolor: 'white', borderRadius: 10, boxShadow: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: 'white' }} />} sx={{ bgcolor: '#444e5e', color: 'white', borderRadius: 1 }}>
            <Typography variant="h5" fontWeight="bold">Vision System for Deep Learning</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="h6" sx={{ color: "#444e5e" }}>
              The Vision System enables deep learning-based hydrological modeling. It allows users to
              design, train, and deploy models with custom configurations to predict hydrological
              variables using satellite, climate, and geospatial datasets.
            </Typography>
          </AccordionDetails>
        </Accordion>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Accordion sx={{ bgcolor: 'white', borderRadius: 10, boxShadow: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: 'white' }} />} sx={{ bgcolor: '#444e5e', color: 'white', borderRadius: 1 }}>
            <Typography variant="h5" fontWeight="bold">HydroGeoDataset</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="h6" sx={{ color: "#444e5e" }}>
              HydroGeoDataset compiles national datasets such as MODIS, PRISM, LOCA2, and Wellogic,
              alongside deep modeling and deep learning-derived data. It provides high-resolution
              hydrological, geological, and climate data for Michigan's Lower Peninsula.
            </Typography>
          </AccordionDetails>
        </Accordion>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Accordion sx={{ bgcolor: 'white', borderRadius: 10, boxShadow: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: 'white' }} />} sx={{ bgcolor: '#444e5e', color: 'white', borderRadius: 1 }}>
            <Typography variant="h5" fontWeight="bold">Modeling & Data Outputs</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="h6" sx={{ color: "#444e5e" }}>
              Users can generate spatiotemporal simulations, time-series predictions, and
              high-resolution GIS outputs in HDS format. The system supports real-time data
              visualization, model comparisons, and downloadable results for further analysis.
            </Typography>
          </AccordionDetails>
        </Accordion>
      </Box>

      {/* Contact & Resources */}
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Typography variant="h4" fontWeight="bold" sx={{ color: "white", mt: 10 }}>
          Contact & Resources
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 2 }}>
          <Button
            variant="contained"
            href="mailto:rafieiva@msu.edu"
            sx={{
              backgroundColor: '#ff8500',
              fontWeight: 'bold',
              width: 'fit-content',
              paddingX: 3,
              '&:hover': {
                backgroundColor: '#e67500',
              }
            }}
          >
            Contact Us
          </Button>
          <Button
            variant="contained"
            href="https://bitbucket.org/vahidrafiei/swatgenx/src/main/"
            target="_blank"
            sx={{
              backgroundColor: '#ff8500',
              fontWeight: 'bold',
              width: 'fit-content',
              paddingX: 3,
              '&:hover': {
                backgroundColor: '#e67500',
              }
            }}
          >
            Source Code Repository
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export default AboutUsTemplate;