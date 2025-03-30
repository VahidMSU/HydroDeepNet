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
  faThLarge,
  faThList,
  faList,
  faSortAmountDown,
  faSortAmountUp,
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
  DirectoryGuide,
  ViewControls,
  ViewButton,
  ViewModeLabel,
  ListViewHeader,
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
  handleDownloadDirectory,
  errorMessage = '',
  isLoading = false,
  viewMode = 'grid-large',
  setViewMode,
  sortOrder = 'name-asc',
  setSortOrder,
}) => {
  const pathParts = currentPath ? currentPath.split('/').filter(Boolean) : [];

  // Sort handler function to prepare for sort buttons
  const handleSortChange = (newSortOrder) => {
    setSortOrder(newSortOrder);
  };

  // Toggle sort direction
  const toggleSortDirection = () => {
    if (sortOrder.endsWith('-asc')) {
      handleSortChange(sortOrder.replace('-asc', '-desc'));
    } else {
      handleSortChange(sortOrder.replace('-desc', '-asc'));
    }
  };

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

      {/* Directory Structure Description - Only shown at root level */}
      {currentPath === '' && (
        <DirectoryGuide>
          <h3>
            <FontAwesomeIcon icon={faInfoCircle} className="guide-icon" />
            Directory Structure Guide
          </h3>
          <p>Models are organized in the following structure:</p>
          <ul>
            <li>
              <strong>VPUID Folders (HUC4 level)</strong> - Top level directories (e.g., 0712, 1605)
            </li>
            <li>
              <strong>huc12 Directory</strong> - Contains folders organized by HUC12 watershed
              clusters
            </li>
            <li>
              <strong>USGS Gauging Station</strong> - Named after the streamflow gauging station
              (e.g., 05536265)
            </li>
            <li>
              <strong>Data Folders</strong> - Inside you'll find:
              <ul>
                <li>
                  <code>streamflow_data</code> - Historical streamflow measurements
                </li>
                <li>
                  <code>PRISM</code> - Meteorological data used in the model
                </li>
                <li>
                  <code>SWAT_MODEL_Web_Application</code> - Contains SWAT+ model files and QGIS
                  project
                </li>
              </ul>
            </li>
          </ul>
          <p className="guide-footer">
            Navigate through the folders to access and download your model files.
          </p>
        </DirectoryGuide>
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

      {/* View Controls */}
      <ViewControls>
        <ViewModeLabel>View:</ViewModeLabel>
        <ViewButton
          onClick={() => setViewMode('grid-large')}
          className={viewMode === 'grid-large' ? 'active' : ''}
          title="Large Grid View"
        >
          <FontAwesomeIcon icon={faThLarge} />
        </ViewButton>
        <ViewButton
          onClick={() => setViewMode('grid-small')}
          className={viewMode === 'grid-small' ? 'active' : ''}
          title="Small Grid View"
        >
          <FontAwesomeIcon icon={faThList} />
        </ViewButton>
        <ViewButton
          onClick={() => setViewMode('list-view')}
          className={viewMode === 'list-view' ? 'active' : ''}
          title="List View"
        >
          <FontAwesomeIcon icon={faList} />
        </ViewButton>

        <span style={{ flex: 1 }}></span>

        <ViewModeLabel>Sort by:</ViewModeLabel>
        <ViewButton
          onClick={() =>
            handleSortChange(
              sortOrder.startsWith('name')
                ? 'date-' + sortOrder.split('-')[1]
                : 'name-' + sortOrder.split('-')[1],
            )
          }
          title={sortOrder.startsWith('name') ? 'Sort by date' : 'Sort by name'}
        >
          {sortOrder.startsWith('name') ? 'Name' : 'Date'}
        </ViewButton>
        <ViewButton
          onClick={toggleSortDirection}
          title={sortOrder.endsWith('asc') ? 'Sort descending' : 'Sort ascending'}
        >
          <FontAwesomeIcon icon={sortOrder.endsWith('asc') ? faSortAmountUp : faSortAmountDown} />
        </ViewButton>
      </ViewControls>

      {/* List View Header - Only show in list view */}
      {viewMode === 'list-view' && (
        <ListViewHeader>
          <div className="name">Name</div>
          <div className="details">
            <span>Size</span>
            <span>Last Modified</span>
            <span>Type</span>
          </div>
          <div className="actions">Actions</div>
        </ListViewHeader>
      )}

      {/* Directory and File Grid */}
      <ContentGrid className={viewMode}>
        {/* Directories */}
        {contents.directories.map((dir, index) => (
          <FolderCard key={`dir-${index}`}>
            {viewMode === 'list-view' ? (
              // List View Layout
              <>
                <div className="item-details">
                  <FolderHeader>
                    <FontAwesomeIcon icon={faFolder} className="icon" />
                    <h3 title={dir.name}>{dir.name}</h3>
                    {dir.items > 0 && <BadgeCount>{dir.items}</BadgeCount>}
                  </FolderHeader>

                  <ItemInfo className="item-metadata">
                    <p>
                      <FontAwesomeIcon icon={faInfoCircle} className="info-icon" />
                      {dir.items} item{dir.items !== 1 ? 's' : ''}
                    </p>
                    <p>
                      <FontAwesomeIcon icon={faCalendarAlt} className="info-icon" />
                      {formatDate(dir.created)}
                    </p>
                    <span>Folder</span>
                  </ItemInfo>
                </div>

                <div className="item-actions">
                  <FolderButton onClick={() => handleDirectoryClick(dir.path)}>
                    <FontAwesomeIcon icon={faFolderOpen} className="icon" />
                    Open
                  </FolderButton>
                  <FileButton
                    onClick={(event) => {
                      event.preventDefault();
                      handleDownloadDirectory(dir.download_zip_url);
                    }}
                    className="icon-only"
                  >
                    <FontAwesomeIcon icon={faDownload} className="icon" />
                  </FileButton>
                </div>
              </>
            ) : (
              // Grid View Layout
              <>
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
                  onClick={(event) => {
                    event.preventDefault();
                    handleDownloadDirectory(dir.download_zip_url);
                  }}
                  className="icon-only"
                >
                  <FontAwesomeIcon icon={faDownload} className="icon" />
                </FileButton>
              </>
            )}
          </FolderCard>
        ))}

        {/* Files - Only show when not at root level */}
        {currentPath !== '' &&
          contents.files.map((file, index) => (
            <FileCard key={`file-${index}`}>
              {viewMode === 'list-view' ? (
                // List View Layout
                <>
                  <div className="item-details">
                    <FileHeader>
                      <FontAwesomeIcon icon={getFileIcon(file.name)} className="icon" />
                      <h3 title={file.name}>{file.name}</h3>
                    </FileHeader>

                    <ItemInfo className="item-metadata">
                      <p>
                        <FontAwesomeIcon icon={faInfoCircle} className="info-icon" />
                        {formatFileSize(file.size)}
                      </p>
                      <p>
                        <FontAwesomeIcon icon={faCalendarAlt} className="info-icon" />
                        {formatDate(file.modified)}
                      </p>
                      <FileTypeIcon className={getFileExtension(file.name) || 'default'}>
                        {getFileExtension(file.name).substring(0, 3)}
                      </FileTypeIcon>
                    </ItemInfo>
                  </div>

                  <div className="item-actions">
                    <FileButton
                      onClick={(event) => {
                        event.preventDefault();
                        handleDownloadFile(file.download_url);
                      }}
                      className="icon-only"
                    >
                      <FontAwesomeIcon icon={faDownload} className="icon" />
                    </FileButton>
                  </div>
                </>
              ) : (
                // Grid View Layout
                <>
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
                      <FileTypeIcon className={getFileExtension(file.name) || 'default'}>
                        {getFileExtension(file.name).substring(0, 3)}
                      </FileTypeIcon>
                    </span>
                  </ItemInfo>

                  <FileButton
                    onClick={(event) => {
                      event.preventDefault();
                      handleDownloadFile(file.download_url);
                    }}
                    className="icon-only"
                  >
                    <FontAwesomeIcon icon={faDownload} className="icon" />
                  </FileButton>
                </>
              )}
            </FileCard>
          ))}

        {/* Empty State - Adjust the condition to account for hidden files at root */}
        {contents.directories.length === 0 && contents.files.length === 0 && !isLoading && (
          <EmptyState>
            <FontAwesomeIcon icon={faFolder} className="icon" />
            <h3>No Files or Folders Found</h3>
            <p>This folder is currently empty</p>
          </EmptyState>
        )}

        {/* Loading state */}
        {isLoading && (
          <EmptyState>
            <div className="loading-spinner"></div>
            <h3>Loading Content...</h3>
            <p>Please wait while we retrieve your files</p>
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
  isLoading: PropTypes.bool,
  viewMode: PropTypes.oneOf(['grid-large', 'grid-small', 'list-view']),
  setViewMode: PropTypes.func.isRequired,
  sortOrder: PropTypes.string,
  setSortOrder: PropTypes.func.isRequired,
};

export default UserDashboardTemplate;
