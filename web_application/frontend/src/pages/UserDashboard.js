import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { library } from '@fortawesome/fontawesome-svg-core';
import {
  faDownload,
  faFolder,
  faFile,
  faFilePdf,
  faFileWord,
  faFileExcel,
  faFileImage,
  faFileArchive,
  faFileCode,
  faFileAlt,
  faThLarge,
  faThList,
  faList,
} from '@fortawesome/free-solid-svg-icons';
import UserDashboardTemplate from '../components/templates/UserDashboard';

// Add icons to the library
library.add(
  faDownload,
  faFolder,
  faFile,
  faFilePdf,
  faFileWord,
  faFileExcel,
  faFileImage,
  faFileArchive,
  faFileCode,
  faFileAlt,
  faThLarge,
  faThList,
  faList,
);

const UserDashboard = () => {
  const [contents, setContents] = useState({ directories: [], files: [] });
  const [currentPath, setCurrentPath] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // New state for view and sort options
  const [viewMode, setViewMode] = useState(() => {
    // Try to load from localStorage, default to 'list-view' (changed from 'grid-large')
    return localStorage.getItem('userDashboardViewMode') || 'list-view';
  });
  const [sortOrder, setSortOrder] = useState(() => {
    // Try to load from localStorage, default to 'name-asc'
    return localStorage.getItem('userDashboardSortOrder') || 'name-asc';
  });

  // Save preferences to localStorage when they change
  useEffect(() => {
    localStorage.setItem('userDashboardViewMode', viewMode);
  }, [viewMode]);

  useEffect(() => {
    localStorage.setItem('userDashboardSortOrder', sortOrder);
  }, [sortOrder]);

  const fetchFiles = async (path = '') => {
    setIsLoading(true);
    try {
      const response = await axios.get('/api/user_files', { params: { subdir: path } });

      // Sort the files and directories based on current sort settings
      const sortedContents = sortContents(response.data, sortOrder);

      setContents(sortedContents);
      setCurrentPath(path);
      setErrorMessage('');
    } catch (error) {
      console.error('Error fetching files:', error);
      setErrorMessage('Failed to fetch files. Please try again later.');
      setContents({ directories: [], files: [] });
    } finally {
      setIsLoading(false);
    }
  };

  // Sort content based on sort order
  const sortContents = (data, order) => {
    const [property, direction] = order.split('-');
    const sortDirection = direction === 'asc' ? 1 : -1;

    // Create deep copies to avoid mutating props
    const sortedDirectories = [...data.directories];
    const sortedFiles = [...data.files];

    // Sort directories
    sortedDirectories.sort((a, b) => {
      if (property === 'name') {
        return sortDirection * a.name.localeCompare(b.name);
      } else if (property === 'date') {
        const dateA = new Date(a.created || 0);
        const dateB = new Date(b.created || 0);
        return sortDirection * (dateA - dateB);
      }
      return 0;
    });

    // Sort files
    sortedFiles.sort((a, b) => {
      if (property === 'name') {
        return sortDirection * a.name.localeCompare(b.name);
      } else if (property === 'date') {
        const dateA = new Date(a.modified || 0);
        const dateB = new Date(b.modified || 0);
        return sortDirection * (dateA - dateB);
      }
      return 0;
    });

    return { directories: sortedDirectories, files: sortedFiles };
  };

  // Re-sort contents when sort order changes
  useEffect(() => {
    if (contents.directories.length > 0 || contents.files.length > 0) {
      setContents((prevContents) => sortContents(prevContents, sortOrder));
    }
  }, [sortOrder, contents.directories.length, contents.files.length]);

  useEffect(() => {
    fetchFiles();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDirectoryClick = (path) => {
    fetchFiles(path);
  };

  const handleDownloadFile = (url) => {
    console.log('Download URL triggered:', url);
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
      handleDownloadDirectory={handleDownloadDirectory}
      errorMessage={errorMessage}
      viewMode={viewMode}
      setViewMode={setViewMode}
      sortOrder={sortOrder}
      setSortOrder={setSortOrder}
    />
  );
};

export default UserDashboard;
