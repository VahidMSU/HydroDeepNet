import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { library } from '@fortawesome/fontawesome-svg-core';
import { faDownload, faFolder, faFile } from '@fortawesome/free-solid-svg-icons';
import UserDashboardTemplate from '../components/templates/UserDashboard';

// Add icons to the library
library.add(faDownload, faFolder, faFile);

const UserDashboard = () => {
  const [contents, setContents] = useState({ directories: [], files: [] });
  const [currentPath, setCurrentPath] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  const fetchFiles = async (path = '') => {
    setIsLoading(true);
    try {
      const response = await axios.get('/api/user_files', { params: { subdir: path } });
      setContents(response.data);
      setCurrentPath(path);
      setErrorMessage('');
    } catch (error) {
      console.error('Error fetching files:', error);
      setErrorMessage('Failed to fetch files.');
      setContents({ directories: [], files: [] });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const handleDirectoryClick = (path) => {
    fetchFiles(path);
  };

  const handleDownloadFile = (url) => {
    console.log('Download URL triggered:', url); // Log the URL
    window.location.href = url; // Trigger the backend route
  };

  const handleDownloadDirectory = (url) => {
    console.log('Download Directory URL triggered:', url);
    window.location.href = url;
  };

  return (
    <UserDashboardTemplate
      contents={contents}
      currentPath={currentPath}
      isLoading={isLoading}
      handleDirectoryClick={handleDirectoryClick}
      handleDownloadFile={handleDownloadFile}
      handleDownloadDirectory={handleDownloadDirectory} // Ensure this is passed
      errorMessage={errorMessage}
    />
  );
};

export default UserDashboard;
