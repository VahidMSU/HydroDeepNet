import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Link,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import SecurityIcon from '@mui/icons-material/Security';
import InfoIcon from '@mui/icons-material/Info';
import VpnKeyIcon from '@mui/icons-material/VpnKey';

const FTPSServer = () => {
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'flex-start',
        minHeight: '100vh',
        p: 3,
      }}
    >
      <Paper
        elevation={3}
        sx={{
          p: 4,
          maxWidth: 800,
          width: '100%',
          bgcolor: '#ffffff',
          borderRadius: 2,
        }}
      >
        <Typography variant="h4" component="h1" gutterBottom>
          FTPS Server Access
        </Typography>

        <Typography variant="body1" paragraph>
          Our FTPS server provides secure file transfer capabilities for authorized users. Below
          you'll find important information about accessing and using the FTPS service.
        </Typography>

        <List>
          <ListItem>
            <ListItemIcon>
              <CloudDownloadIcon />
            </ListItemIcon>
            <ListItemText
              primary="Server Information"
              secondary="ftps://ftps.hydrodeepnet.org:21"
            />
          </ListItem>

          <ListItem>
            <ListItemIcon>
              <SecurityIcon />
            </ListItemIcon>
            <ListItemText
              primary="Security"
              secondary="All transfers are encrypted using TLS 1.2 or higher for maximum security"
            />
          </ListItem>

          <ListItem>
            <ListItemIcon>
              <VpnKeyIcon />
            </ListItemIcon>
            <ListItemText
              primary="Authentication"
              secondary="Use your HydroDeepNet account credentials to access the FTPS server"
            />
          </ListItem>

          <ListItem>
            <ListItemIcon>
              <InfoIcon />
            </ListItemIcon>
            <ListItemText
              primary="Support"
              secondary="For technical assistance, please contact our support team through the Contact page"
            />
          </ListItem>
        </List>

        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            Recommended FTPS Clients
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="FileZilla"
                secondary={
                  <Link
                    href="https://filezilla-project.org/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Download FileZilla
                  </Link>
                }
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="WinSCP (Windows)"
                secondary={
                  <Link href="https://winscp.net/" target="_blank" rel="noopener noreferrer">
                    Download WinSCP
                  </Link>
                }
              />
            </ListItem>
          </List>
        </Box>

        <Box sx={{ mt: 4 }}>
          <Typography variant="body2" color="text.secondary">
            For more information about using the FTPS server, please refer to our documentation or
            contact support.
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default FTPSServer;
