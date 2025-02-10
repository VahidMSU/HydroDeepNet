import React from 'react';
import UserFilesForm from '../forms/UserFiles.js';
import {
  UserDashboardContainer,
  Title,
  DirectoryList,
  FileList,
  ErrorMessage,
} from '../../styles/UserDashboard.tsx'; // Import styled components

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
      {errorMessage && <ErrorMessage>{errorMessage}</ErrorMessage>}
      <DirectoryList>{/* Directory list content */}</DirectoryList>
      <FileList>{/* File list content */}</FileList>
    </UserDashboardContainer>
  );
};

export default UserDashboardTemplate;
