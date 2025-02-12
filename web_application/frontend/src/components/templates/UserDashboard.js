import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import PropTypes from 'prop-types';
import {
  DashboardContainer,
  DashboardHeader,
  ContentGrid,
  FileCard,
  FileHeader,
  FileInfo,
  ActionButton,
  BreadcrumbNav,
  EmptyState,
  ErrorMessage,
} from '../../styles/UserDashboard.tsx';

const UserDashboardTemplate = ({
  contents = { directories: [], files: [] },
  currentPath = '',
  handleDirectoryClick,
  handleDownloadFile,
  handleDownloadDirectory,
  errorMessage,
}) => {
  // Only split the path if it exists and is not empty
  const pathParts = currentPath ? currentPath.split('/').filter(Boolean) : [];

  return (
    <DashboardContainer>
      <DashboardHeader>
        <h1>User Dashboard</h1>
      </DashboardHeader>

      {errorMessage && <ErrorMessage>{errorMessage}</ErrorMessage>}

      <BreadcrumbNav>
        <button onClick={() => handleDirectoryClick('')}>Home</button>
        {pathParts.length > 0 &&
          pathParts.map((part, index) => (
            <React.Fragment key={index}>
              <span>/</span>
              <button onClick={() => handleDirectoryClick(pathParts.slice(0, index + 1).join('/'))}>
                {part}
              </button>
            </React.Fragment>
          ))}
      </BreadcrumbNav>

      <ContentGrid>
        {contents.directories.map((dir, index) => (
          <FileCard key={`dir-${index}`}>
            <FileHeader>
              <FontAwesomeIcon icon="folder" className="icon" />
              <h3>{dir.name}</h3>
            </FileHeader>
            <FileInfo>
              <p>Created: {new Date(dir.created).toLocaleDateString()}</p>
              <p>{dir.items} items</p>
            </FileInfo>
            <ActionButton onClick={() => handleDirectoryClick(dir.path)}>
              <FontAwesomeIcon icon="folder" className="icon" /> Open
            </ActionButton>
          </FileCard>
        ))}

        {contents.files.map((file, index) => (
          <FileCard key={`file-${index}`}>
            <FileHeader>
              <FontAwesomeIcon icon="file" className="icon" />
              <h3>{file.name}</h3>
            </FileHeader>
            <FileInfo>
              <p>Size: {file.size}</p>
              <p>Modified: {new Date(file.modified).toLocaleDateString()}</p>
            </FileInfo>
            <ActionButton onClick={() => handleDownloadFile(file.download_url)}>
              <FontAwesomeIcon icon="download" className="icon" /> Download
            </ActionButton>
          </FileCard>
        ))}

        {contents.directories.length === 0 && contents.files.length === 0 && (
          <EmptyState>
            <FontAwesomeIcon icon="folder-open" className="icon" />
            <h3>No Files Found</h3>
            <p>This folder is empty</p>
          </EmptyState>
        )}
      </ContentGrid>
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
  handleDownloadDirectory: PropTypes.func.isRequired,
  errorMessage: PropTypes.string,
};

// Add default props
UserDashboardTemplate.defaultProps = {
  contents: { directories: [], files: [] },
  currentPath: '',
  errorMessage: '',
};

export default UserDashboardTemplate;
