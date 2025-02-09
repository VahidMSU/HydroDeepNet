import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faDownload, faFolder, faFile } from '@fortawesome/free-solid-svg-icons';
import {
  FileContainer,
  DirectoryButton,
  DownloadButton,
  ErrorMessage,
  GoUpButton,
  DirectoryContainer,
  FileList,
  DirectoryItem,
  FileItem,
} from '../../styles/UserFilesStyles.tsx';

const UserFilesForm = ({
  contents,
  handleDirectoryClick,
  handleDownloadFile,
  handleDownloadDirectory,
  errorMessage,
}) => {
  return (
    <FileContainer>
      {errorMessage && <ErrorMessage>{errorMessage}</ErrorMessage>}
      {contents.parent_path && (
        <GoUpButton onClick={() => handleDirectoryClick(contents.parent_path)}>Go Up</GoUpButton>
      )}
      <DirectoryContainer>
        {contents.directories.map((directory, index) => (
          <DirectoryItem key={`directory-${index}`}>
            <DirectoryButton onClick={() => handleDirectoryClick(directory.path)}>
              <FontAwesomeIcon icon={faFolder} /> {directory.name}
            </DirectoryButton>
            <DownloadButton onClick={() => handleDownloadDirectory(directory.download_zip_url)}>
              <FontAwesomeIcon icon={faDownload} />
            </DownloadButton>
          </DirectoryItem>
        ))}
      </DirectoryContainer>
      <FileList>
        {contents.files.map((file, index) => (
          <FileItem key={`file-${index}`}>
            <span><FontAwesomeIcon icon={faFile} /> {file.name}</span>
            <DownloadButton onClick={() => handleDownloadFile(file.download_url)}>
              <FontAwesomeIcon icon={faDownload} />
            </DownloadButton>
          </FileItem>
        ))}
      </FileList>
    </FileContainer>
  );
};

export default UserFilesForm;
