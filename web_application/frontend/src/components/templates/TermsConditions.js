import React from 'react';
import { Box, Typography, Paper, Button, List, ListItem, ListItemText, Link } from '@mui/material';
import { styled } from '@mui/system';

// Styled Components for Consistency
const TermsContainer = styled(Box)({
  bgcolor: '#2b2b2c',
  minHeight: '100vh',
  px: 4,
  py: 3,
});

const TermsTitle = styled(Typography)({
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

const TermsText = styled(Typography)({
  color: 'white',
  fontSize: '1.1rem',
  lineHeight: 1.6,
  marginBottom: '1.5rem',
});

const ContactSection = styled(Box)({
  padding: '2rem',
  textAlign: 'center',
});

// Terms and Conditions Component
const TermsAndConditionsTemplate = () => {
  return (
    <TermsContainer>
      <TermsTitle>Terms and Conditions</TermsTitle>

      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <SectionHeader>Overview</SectionHeader>
        <TermsText>
          This web application is provided as a research collaboration tool for hydrological and
          environmental modeling. By accessing and using this platform, you agree to the following
          terms.
        </TermsText>
      </Paper>

      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <SectionHeader>Data Sources and Technology</SectionHeader>
        <TermsText>
          The application is built on open-source technologies and integrates datasets from public
          sources, including:
        </TermsText>
        <List sx={{ pl: 3, color: 'white', listStyleType: 'disc' }}>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText primary="National Solar Radiation Database (NSRDB)" sx={{ color: 'white' }} />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText primary="NHDPlus High Resolution" sx={{ color: 'white' }} />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText primary="LANDFIRE and STATSGO2" sx={{ color: 'white' }} />
          </ListItem>
          <ListItem sx={{ display: 'list-item', py: 0.3 }}>
            <ListItemText primary="SNODAS hydrological datasets" sx={{ color: 'white' }} />
          </ListItem>
        </List>
      </Paper>

      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <SectionHeader>User Responsibilities</SectionHeader>
        <TermsText>
          Users are responsible for ensuring that their activities comply with Michigan State
          University's IT policies and relevant research agreements. Unauthorized access, data
          scraping, or any form of misuse, including attempts to bypass authentication or disrupt
          system functionality, is strictly prohibited.
        </TermsText>
      </Paper>

      <Paper sx={{ bgcolor: '#444e5e', p: 3, my: 3, borderRadius: 2 }}>
        <SectionHeader>Data Usage</SectionHeader>
        <TermsText>
          Access to specific datasets may be subject to licensing agreements or institutional
          policies. Users contributing data must ensure they have the appropriate permissions to
          share it on this platform.
        </TermsText>
      </Paper>

      <SectionHeader sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 5, mb: 3 }}>
        Contact & Resources
      </SectionHeader>

      {/* Buttons Section */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 1, mb: 4 }}>
        <Button variant="contained" sx={{ bgcolor: '#ff8500' }} href="mailto:rafieiva@msu.edu">
          Contact Us
        </Button>
        <Button variant="contained" sx={{ bgcolor: '#ff8500' }} onClick={() => alert('Learn More clicked!')}>
          Learn More
        </Button>
        <Button variant="contained" sx={{ bgcolor: '#ff8500' }} href="https://msu.edu/terms">
          MSU Terms of Use
        </Button>
        <Button variant="contained" sx={{ bgcolor: '#ff8500' }} onClick={() => alert('Download Terms clicked!')}>
          Download Terms
        </Button>
      </Box>
    </TermsContainer>
  );
};

export default TermsAndConditionsTemplate;