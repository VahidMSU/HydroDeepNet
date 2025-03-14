import React from 'react';
import { Box, Typography, Paper, Button, List, ListItem, ListItemText } from '@mui/material';
import { styled } from '@mui/system';

// Styled Components for Consistency
const PrivacyContainer = styled(Box)({
  bgcolor: '#2b2b2c',
  minHeight: '100vh',
  px: 4,
  py: 3,
});

const PrivacyTitle = styled(Typography)({
  fontSize: '3.5rem',
  fontWeight: 'bold',
  color: 'white',
  textAlign: 'center',
  marginBottom: '2rem',
});

const SectionHeader = styled(Typography)({
  fontSize: '1.8rem',
  fontWeight: 'bold',
  color: '#ff8500',
  marginBottom: '1rem',
});

const PolicyText = styled(Typography)({
  color: 'white',
  fontSize: '1.1rem',
  lineHeight: 1.6,
  marginBottom: '1.5rem',
});

// Privacy Template Component
const PrivacyTemplate = () => {
  return (
    <PrivacyContainer>
      <PrivacyTitle>Privacy Policy</PrivacyTitle>

      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <SectionHeader>Our Commitment to Privacy</SectionHeader>
        <PolicyText>
          This web application is designed to support research and collaboration by providing access
          to hydrological modeling tools and datasets. We respect user privacy and are committed to
          protecting any information collected during platform use.
        </PolicyText>
      </Paper>

      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <SectionHeader>Data Collection and Use</SectionHeader>
        <PolicyText>
          We collect user interactions with the application solely for improving system
          functionality, security, and performance. This includes authentication logs and feature
          usage analytics but excludes personally identifiable data beyond what is necessary for
          secure access. All passwords are stored securely using encryption standards.
        </PolicyText>
      </Paper>

      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <SectionHeader>Data Sharing</SectionHeader>
        <PolicyText>
          User data will not be shared with third parties unless mandated by Michigan State
          University's IT security policies or legal requirements. Any data provided for research
          purposes remains under the ownership of the contributing institution or researcher, and
          its use follows applicable agreements.
        </PolicyText>
      </Paper>

      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <SectionHeader>Data Sources</SectionHeader>
        <PolicyText>
          The platform operates entirely on open-source software and integrates national datasets,
          including:
        </PolicyText>
        <List sx={{ pl: 3, color: 'white', listStyleType: 'disc' }}>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText
              primary="National Solar Radiation Database (NSRDB)"
              sx={{ color: 'white' }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText primary="NHDPlus and LANDFIRE data" sx={{ color: 'white' }} />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText primary="3D Elevation Program and STATSGO2" sx={{ color: 'white' }} />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText primary="SNODAS hydrological data" sx={{ color: 'white' }} />
          </ListItem>
        </List>
      </Paper>

      <Paper sx={{ bgcolor: '#444e5e', p: 3, mt: 3, borderRadius: 2 }}>
        <SectionHeader>Acceptance of Terms</SectionHeader>
        <PolicyText>
          By using this application, you acknowledge that minimal system interaction data is
          collected to improve platform reliability. You may contact the administrators for
          questions regarding data policies or security measures.
        </PolicyText>
      </Paper>

      <SectionHeader sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 5, mb: 3 }}>
        Contact & Resources
      </SectionHeader>

      {/* Buttons Section */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 1, mb: 4 }}>
        <Button variant="contained" sx={{ bgcolor: '#ff8500' }} href="mailto:rafieiva@msu.edu">
          Contact Us
        </Button>
        <Button
          variant="contained"
          sx={{ bgcolor: '#ff8500' }}
          onClick={() => alert('Learn More clicked!')}
        >
          Learn More
        </Button>
        <Button variant="contained" sx={{ bgcolor: '#ff8500' }} href="https://msu.edu/privacy">
          MSU Privacy Policy
        </Button>
        <Button
          variant="contained"
          sx={{ bgcolor: '#ff8500' }}
          onClick={() => alert('Download Policy clicked!')}
        >
          Download Policy
        </Button>
      </Box>
    </PrivacyContainer>
  );
};

export default PrivacyTemplate;
