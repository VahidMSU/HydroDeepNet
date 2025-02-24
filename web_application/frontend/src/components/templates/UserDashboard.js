import React from 'react';
import PropTypes from 'prop-types';
import { Box, Typography, Grid2, Card, CardContent, Button, Breadcrumbs, Link, Alert } from '@mui/material';
import { styled } from '@mui/system';
import FolderIcon from '@mui/icons-material/Folder';
import FilePresentIcon from '@mui/icons-material/InsertDriveFile';
import DownloadIcon from '@mui/icons-material/Download';

// Styled Components
const DashboardContainer = styled(Box)({
  maxWidth: '1000px',
  margin: '2rem auto',
  padding: '2rem',
  backgroundColor: '#444e5e',
  borderRadius: '16px',
  boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)',
});

const DashboardHeader = styled(Typography)({
  color: 'white',
  fontSize: '2.5rem',
  textAlign: 'center',
  marginBottom: '1rem',
});

const FileCard = styled(Card)({
  backgroundColor: '#ffffff',
  borderRadius: '8px',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-3px)',
    boxShadow: '0 5px 15px rgba(255, 133, 0, 0.3)',
  },
});

const FileHeader = styled(Typography)({
  display: 'flex',
  alignItems: 'center',
  fontSize: '1.2rem',
  fontWeight: 600,
  color: '#2b2b2c',
});

const FileInfo = styled(Typography)({
  fontSize: '0.9rem',
  color: '#555',
});

const ActionButton = styled(Button)({
  backgroundColor: '#e67500',
  color: '#ffffff',
  padding: '0.5rem 1rem',
  fontSize: '1rem',
  fontWeight: 600,
  '&:hover': {
    backgroundColor: '#ff8500',
  },
});

const EmptyState = styled(Box)({
  textAlign: 'center',
  color: '#ffffff',
  marginTop: '2rem',
});

// UserDashboard Component
const UserDashboardTemplate = ({
  contents = { directories: [], files: [] },
  currentPath = '',
  handleDirectoryClick,
  handleDownloadFile,
  errorMessage,
}) => {
  const pathParts = currentPath ? currentPath.split('/').filter(Boolean) : [];

  return (
    <DashboardContainer>
      <DashboardHeader variant="h1" sx={{ fontWeight: 'bold', mb: 3}}>User Dashboard</DashboardHeader>

      {/* Error Message */}
      {errorMessage && <Alert severity="error">{errorMessage}</Alert>}

      {/* Breadcrumb Navigation */}
      <Breadcrumbs sx={{ color: 'white' }}>
        <Link
          component="button"
          underline="hover"
          variant='h4'
          sx={{ color: '#ffa533', fontWeight: 'bold', mt: 3 }}
          onClick={() => handleDirectoryClick('')}
        >
          Home
        </Link>
        {pathParts.map((part, index) => (
          <Link
            key={index}
            component="button"
            underline="hover"
            sx={{ color: '#ffffff' }}
            onClick={() => handleDirectoryClick(pathParts.slice(0, index + 1).join('/'))}
          >
            {part}
          </Link>
        ))}
      </Breadcrumbs>

      {/* Directory and File Grid */}
      <Grid2 container spacing={3} justifyContent="center" alignItems="center">
        {/* Directories */}
        {contents.directories.map((dir, index) => (
          <Grid2 item xs={12} sm={6} md={4} key={`dir-${index}`}>
            <FileCard>
              <CardContent>
                <FileHeader>
                  <FolderIcon sx={{ color: '#ffa533', mr: 1 }} />
                  {dir.name}
                </FileHeader>
                <FileInfo>Created: {new Date(dir.created).toLocaleDateString()}</FileInfo>
                <FileInfo>{dir.items} items</FileInfo>
                <ActionButton fullWidth onClick={() => handleDirectoryClick(dir.path)}>
                  Open Folder
                </ActionButton>
              </CardContent>
            </FileCard>
          </Grid2>
        ))}

        {/* Files */}
        {contents.files.map((file, index) => (
          <Grid2 item xs={12} sm={6} md={4} key={`file-${index}`}>
            <FileCard>
              <CardContent>
                <FileHeader>
                  <FilePresentIcon sx={{ color: '#2b2b2c', mr: 1 }} />
                  {file.name}
                </FileHeader>
                <FileInfo>Size: {file.size}</FileInfo>
                <FileInfo>Modified: {new Date(file.modified).toLocaleDateString()}</FileInfo>
                <ActionButton fullWidth onClick={() => handleDownloadFile(file.download_url)}>
                  <DownloadIcon sx={{ mr: 1 }} /> Download
                </ActionButton>
              </CardContent>
            </FileCard>
          </Grid2>
        ))}

        {/* Empty State */}
        {contents.directories.length === 0 && contents.files.length === 0 && (
          <Grid2 item xs={12}>
            <EmptyState>
              <FolderIcon sx={{ fontSize: 60, color: '#ffa533' }} />
              <Typography variant="h5">No Files Found</Typography>
              <Typography>This folder is empty</Typography>
            </EmptyState>
          </Grid2>
        )}
      </Grid2>
    </DashboardContainer>
  );
};

// Add PropTypes for type checking
UserDashboardTemplate.propTypes = {
  contents: PropTypes.shape({
    directories: PropTypes.array,
    files: PropTypes.array,
  }),
  currentPath: PropTypes.string,
  handleDirectoryClick: PropTypes.func.isRequired,
  handleDownloadFile: PropTypes.func.isRequired,
  errorMessage: PropTypes.string,
};

// Add default props
UserDashboardTemplate.defaultProps = {
  contents: { directories: [], files: [] },
  currentPath: '',
  errorMessage: '',
};

export default UserDashboardTemplate;