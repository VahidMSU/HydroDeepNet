import React from 'react';
import { Box, Toolbar, Typography } from '@mui/material';
import VisualizationsDashboardTemplate from '../components/templates/VisualizationsDashboard.js';

const VisualizationsDashboard = () => {
  return (
    <Box sx={{ display: 'flex' }}>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        <Typography variant="h4" sx={{ mb: 2 }}>
          Visualizations Dashboard
        </Typography>
        <VisualizationsDashboardTemplate />
      </Box>
    </Box>
  );
};

export default VisualizationsDashboard;
