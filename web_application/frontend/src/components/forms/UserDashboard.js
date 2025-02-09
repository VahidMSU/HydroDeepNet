import React from 'react';
import { List, ListItem, ListItemText, Button, Typography } from '@mui/material';

const UserFilesForm = ({
  contents = { directories: [], files: [] },
  handleDirectoryClick,
  handleDownloadFile,
  handleDownloadDirectory,
  errorMessage,
}) => {
  return (
    <div>
      {errorMessage && <Typography color="error">{errorMessage}</Typography>}
      <List>
        {contents.directories.length > 0 ? (
          contents.directories.map((directory) => (
            <ListItem
              button
              key={directory.path}
              onClick={() => handleDirectoryClick(directory.path)}
            >
              <ListItemText primary={directory.name} />
            </ListItem>
          ))
        ) : (
          <Typography>No directories found</Typography>
        )}
        {contents.files.length > 0 ? (
          contents.files.map((file) => (
            <ListItem key={file.path}>
              <ListItemText primary={file.name} />
              <Button onClick={() => handleDownloadFile(file.download_url)}>Download</Button>
            </ListItem>
          ))
        ) : (
          <Typography>No files found</Typography>
        )}
      </List>
    </div>
  );
};

export default UserFilesForm;
