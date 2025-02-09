import React from 'react';
import UserFilesForm from '../forms/UserFiles.js';
import '../../styles/UserDashboard.tsx'; // Adjust the path if necessary

const UserDashboardTemplate = ({
  contents,
  handleDirectoryClick,
  handleDownloadFile,
  handleDownloadDirectory,
  errorMessage,
}) => {
  return (
    <main className="container my-5">
      <h1 className="text-center mb-4">User Dashboard</h1>
      <UserFilesForm
        contents={contents}
        handleDirectoryClick={handleDirectoryClick}
        handleDownloadFile={handleDownloadFile}
        handleDownloadDirectory={handleDownloadDirectory}
        errorMessage={errorMessage}
      />
    </main>
  );
};

export default UserDashboardTemplate;
