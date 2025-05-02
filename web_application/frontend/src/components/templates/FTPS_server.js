import React from 'react';
import { Box, Typography, Paper, List, ListItem, ListItemText, Divider } from '@mui/material';

const FTPS_Server = () => {
  return (
    <Paper
      elevation={3}
      sx={{ p: 3, maxWidth: '800px', margin: '0 auto', backgroundColor: '#f8f9fa' }}
    >
      <Typography variant="h4" gutterBottom>
        FTPS Server Access
      </Typography>

      <Typography variant="body1" paragraph>
        You can access SWAT model data via our secure FTPS server. Please use the following
        connection details:
      </Typography>

      <Box sx={{ backgroundColor: '#e9ecef', p: 2, borderRadius: 1, mb: 3 }}>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Server:"
              secondary="swatgenx.com (IP: 207.180.226.103)"
              primaryTypographyProps={{ fontWeight: 'bold' }}
            />
          </ListItem>
          <Divider component="li" />
          <ListItem>
            <ListItemText
              primary="Port:"
              secondary="990 (FTPS)"
              primaryTypographyProps={{ fontWeight: 'bold' }}
            />
          </ListItem>
          <Divider component="li" />
          <ListItem>
            <ListItemText
              primary="Protocol:"
              secondary="FTPS (FTP over SSL/TLS)"
              primaryTypographyProps={{ fontWeight: 'bold' }}
            />
          </ListItem>
          <Divider component="li" />
          <ListItem>
            <ListItemText
              primary="Mode:"
              secondary="Passive"
              primaryTypographyProps={{ fontWeight: 'bold' }}
            />
          </ListItem>
          <Divider component="li" />
          <ListItem>
            <ListItemText
              primary="Authentication:"
              secondary="Username and password (Contact administrator for credentials)"
              primaryTypographyProps={{ fontWeight: 'bold' }}
            />
          </ListItem>
          <Divider component="li" />
          <ListItem>
            <ListItemText
              primary="Directory:"
              secondary="/SWATplus_by_VPUID/"
              primaryTypographyProps={{ fontWeight: 'bold' }}
            />
          </ListItem>
        </List>
      </Box>

      <Typography variant="h6" gutterBottom>
        Recommended FTP Clients
      </Typography>
      <List>
        <ListItem>
          <ListItemText primary="FileZilla" secondary="Cross-platform (Windows, macOS, Linux)" />
        </ListItem>
        <ListItem>
          <ListItemText primary="WinSCP" secondary="Windows only" />
        </ListItem>
        <ListItem>
          <ListItemText primary="Cyberduck" secondary="macOS and Windows" />
        </ListItem>
      </List>

      <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
        Connection Instructions
      </Typography>
      <Typography variant="body2" paragraph>
        1. Open your FTP client
      </Typography>
      <Typography variant="body2" paragraph>
        2. Create a new site/connection with the details above
      </Typography>
      <Typography variant="body2" paragraph>
        3. Ensure you select "Require explicit FTP over TLS" or similar option
      </Typography>
      <Typography variant="body2" paragraph>
        4. Use the provided username and password
      </Typography>
      <Typography variant="body2" paragraph>
        5. Connect using passive mode
      </Typography>

      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 3 }}>
        Note: Access is restricted to the MSU campus network or via VPN. If you're having trouble
        connecting, please ensure you're on the MSU network or connected to the MSU VPN.
      </Typography>
    </Paper>
  );
};

export default FTPS_Server;
