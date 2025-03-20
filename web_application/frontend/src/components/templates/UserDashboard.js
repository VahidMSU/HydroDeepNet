import React from 'react';
import PropTypes from 'prop-types';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faFolder,
  faFolderOpen,
  faFileAlt,
  faFilePdf,
  faFileWord,
  faFileExcel,
  faFileImage,
  faFileArchive,
  faFileCode,
  faFile,
  faDownload,
  faHome,
  faCalendarAlt,
  faInfoCircle,
  faExclamationTriangle,
  faChevronRight,
} from '@fortawesome/free-solid-svg-icons';

import {
  DashboardContainer,
  DashboardHeader,
  ContentGrid,
  FolderCard,
  FileCard,
  FolderHeader,
  FileHeader,
  ItemInfo,
  FolderButton,
  FileButton,
  BreadcrumbNav,
  EmptyState,
  ErrorMessage,
  FileTypeIcon,
  BadgeCount,
} from '../../styles/UserDashboard.tsx';

// Helper function to determine file type icon
const getFileIcon = (fileName) => {
  if (!fileName) return faFile;

  const extension = fileName.split('.').pop().toLowerCase();

  switch (extension) {
    case 'pdf':
      return faFilePdf;
    case 'doc':
    case 'docx':
      return faFileWord;
    case 'xls':
    case 'xlsx':
    case 'csv':
      return faFileExcel;
    case 'jpg':
    case 'jpeg':
    case 'png':
    case 'gif':
    case 'bmp':
      return faFileImage;
    case 'zip':
    case 'rar':
    case 'tar':
    case 'gz':
      return faFileArchive;
    case 'js':
    case 'html':
    case 'css':
    case 'py':
    case 'java':
    case 'cpp':
      return faFileCode;
    default:
      return faFileAlt;
  }
};

// Helper to get file extension for the badge
const getFileExtension = (fileName) => {
  if (!fileName) return 'file';
  const extension = fileName.split('.').pop().toLowerCase();
  return extension || 'file';
};

// Helper function to format file size
const formatFileSize = (bytes) => {
  if (!bytes) return '0 Bytes';

  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));

  return `${parseFloat((bytes / Math.pow(1024, i)).toFixed(2))} ${sizes[i]}`;
};

// Helper function to format date
const formatDate = (dateString) => {
  if (!dateString) return 'Unknown';

  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const UserDashboardTemplate = ({
  contents = { directories: [], files: [] },
  currentPath = '',
  handleDirectoryClick,
  handleDownloadFile,
  handleDownloadDirectory, // Add this line
  errorMessage = '',
}) => {
  const pathParts = currentPath ? currentPath.split('/').filter(Boolean) : [];

  return (
    <DashboardContainer>
      <DashboardHeader>
        <h1>User Dashboard</h1>
      </DashboardHeader>

      {/* Error Message */}
      {errorMessage && (
        <ErrorMessage>
          <FontAwesomeIcon icon={faExclamationTriangle} className="error-icon" />
          <span>{errorMessage}</span>
        </ErrorMessage>
      )}

      {/* Breadcrumb Navigation */}
      <BreadcrumbNav>
        <button onClick={() => handleDirectoryClick('')}>
          <FontAwesomeIcon icon={faHome} className="home-icon" />
          Home
        </button>

        {pathParts.map((part, index) => (
          <React.Fragment key={`path-${index}`}>
            <span className="separator">
              <FontAwesomeIcon icon={faChevronRight} />
            </span>
            <button
              className={index === pathParts.length - 1 ? 'current' : ''}
              onClick={() => handleDirectoryClick(pathParts.slice(0, index + 1).join('/'))}
            >
              {index === pathParts.length - 1 && <FontAwesomeIcon icon={faFolderOpen} />}
              {part}
            </button>
          </React.Fragment>
        ))}
      </BreadcrumbNav>

      {/* Directory and File Grid */}
      <ContentGrid>
        {/* Directories */}
        {contents.directories.map((dir, index) => (
          <FolderCard key={`dir-${index}`}>
            <FolderHeader>
              <FontAwesomeIcon icon={faFolder} className="icon" />
              <h3 title={dir.name}>{dir.name}</h3>
              {dir.items > 0 && <BadgeCount>{dir.items}</BadgeCount>}
            </FolderHeader>

            <ItemInfo>
              <p>
                <FontAwesomeIcon icon={faCalendarAlt} className="info-icon" />
                Created: {formatDate(dir.created)}
              </p>
              <p>
                <FontAwesomeIcon icon={faInfoCircle} className="info-icon" />
                {dir.items} item{dir.items !== 1 ? 's' : ''}
              </p>
            </ItemInfo>

            <FolderButton onClick={() => handleDirectoryClick(dir.path)}>
              <FontAwesomeIcon icon={faFolderOpen} className="icon" />
              Open Folder
            </FolderButton>
            <FileButton
              as="a"
              href={dir.download_zip_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(event) => {
                event.preventDefault();
                console.log('Download directory URL:', dir.download_zip_url);
                handleDownloadDirectory(dir.download_zip_url);
              }}
            >
              <FontAwesomeIcon icon={faDownload} className="icon" />
              Download as ZIP
            </FileButton>
          </FolderCard>
        ))}

        {/* Files */}
        {contents.files.map((file, index) => (
          <FileCard key={`file-${index}`}>
            <FileHeader>
              <FontAwesomeIcon icon={getFileIcon(file.name)} className="icon" />
              <h3 title={file.name}>{file.name}</h3>
            </FileHeader>

            <ItemInfo>
              <p>
                <FontAwesomeIcon icon={faInfoCircle} className="info-icon" />
                Size: {formatFileSize(file.size)}
              </p>
              <p>
                <FontAwesomeIcon icon={faCalendarAlt} className="info-icon" />
                Modified: {formatDate(file.modified)}
              </p>
              <span>
                {' '}
                {/* Replace <p> with <span> to avoid nesting issues */}
                <FileTypeIcon className={getFileExtension(file.name) || 'default'}>
                  {getFileExtension(file.name).substring(0, 3)}
                </FileTypeIcon>
              </span>
            </ItemInfo>

            <FileButton
              as="button" // Change to a button to avoid default anchor behavior
              onClick={(event) => {
                event.preventDefault(); // Prevent default behavior
                event.stopPropagation(); // Prevent event bubbling
                console.log('FileButton clicked, download URL:', file.download_url); // Add logging
                handleDownloadFile(file.download_url); // Trigger the download
              }}
            >
              <FontAwesomeIcon icon={faDownload} className="icon" />
              Download
            </FileButton>
          </FileCard>
        ))}

        {/* Empty State */}
        {contents.directories.length === 0 && contents.files.length === 0 && (
          <EmptyState>
            <FontAwesomeIcon icon={faFolder} className="icon" />
            <h3>No Files or Folders Found</h3>
            <p>This folder is currently empty</p>
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
  handleDownloadDirectory: PropTypes.func.isRequired, // Add this line
  errorMessage: PropTypes.string,
};

export default UserDashboardTemplate;
