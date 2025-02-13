import React from 'react';
import { Box, Typography, Container } from '@mui/material';

const TermsAndConditionsTemplate = () => {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: '#1a1a1a',
        color: 'white',
        py: 4,
      }}
    >
      <Container maxWidth="md">
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Typography variant="h2" component="h1" gutterBottom>
            Terms and Conditions
          </Typography>
        </Box>

        <Box
          sx={{
            bgcolor: 'rgba(255, 255, 255, 0.1)',
            p: 4,
            borderRadius: 2,
          }}
        >
          <Typography paragraph>
            This web application is provided as a research collaboration tool for hydrological and
            environmental modeling. By accessing and using this platform, you agree to the following
            terms.
          </Typography>

          <Typography paragraph>
            The application is built on open-source technologies and integrates datasets from public
            sources, including the National Solar Radiation Database (NSRDB), NHDPlus High
            Resolution, LANDFIRE, STATSGO2, and SNODAS. The platform also utilizes software such as
            QSWAT+, SWATPlusEditor, SWAT+, MODFLOW, and FloPy for modeling workflows.
          </Typography>

          <Typography paragraph>
            Users are responsible for ensuring that their activities comply with Michigan State
            University's IT policies and relevant research agreements. Unauthorized access, data
            scraping, or any form of misuse, including attempts to bypass authentication or disrupt
            system functionality, is strictly prohibited.
          </Typography>

          <Typography paragraph>
            Access to specific datasets may be subject to licensing agreements or institutional
            policies. Users contributing data must ensure they have the appropriate permissions to
            share it on this platform.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default TermsAndConditionsTemplate;
