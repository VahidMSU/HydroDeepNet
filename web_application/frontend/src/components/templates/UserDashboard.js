import React from 'react';
import UserFilesForm from '../forms/UserFiles.js';
import { UserDashboardContainer, Title } from '../../styles/UserDashboard.tsx'; // Import styled components

const UserDashboardTemplate = ({
  contents,
  handleDirectoryClick,
  handleDownloadFile,
  handleDownloadDirectory,
  errorMessage,
}) => {
  return (
    <UserDashboardContainer>
      <Title>User Dashboard</Title>
      <UserFilesForm
        contents={contents}
        handleDirectoryClick={handleDirectoryClick}
        handleDownloadFile={handleDownloadFile}
        handleDownloadDirectory={handleDownloadDirectory}
        errorMessage={errorMessage}
      />
    </UserDashboardContainer>
  );
};

export default UserDashboardTemplate;
