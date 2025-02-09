import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { library } from '@fortawesome/fontawesome-svg-core';
import { faDownload, faFolder, faFile } from '@fortawesome/free-solid-svg-icons';
import UserDashboardTemplate from '../components/templates/UserDashboard'; // Import the new UserDashboardTemplate component

// Add icons to the library
library.add(faDownload, faFolder, faFile);

const UserDashboard = () => {
  const [contents, setContents] = useState({ directories: [], files: [] });
  const [currentPath, setCurrentPath] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const fetchFiles = async (path = '') => {
    try {
      const response = await axios.get('/api/user_files', { params: { subdir: path } });
      setContents(response.data);
      setCurrentPath(path);
    } catch (error) {
      console.error('Error fetching files:', error);
      setErrorMessage('Failed to fetch files.');
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const handleDirectoryClick = (path) => {
    fetchFiles(path);
  };

  const handleDownloadFile = (url) => {
    window.location.href = url;
  };

  const handleDownloadDirectory = (url) => {
    window.location.href = url;
  };

  return (
    <UserDashboardTemplate
      contents={contents}
      handleDirectoryClick={handleDirectoryClick}
      handleDownloadFile={handleDownloadFile}
      handleDownloadDirectory={handleDownloadDirectory}
      errorMessage={errorMessage}
    />
  );
};

export default UserDashboard;
