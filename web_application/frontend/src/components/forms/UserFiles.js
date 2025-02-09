import React from 'react';

const UserFilesForm = ({
  contents,
  handleDirectoryClick,
  handleDownloadFile,
  handleDownloadDirectory,
  errorMessage,
}) => {
  return (
    <div>
      {errorMessage && <div className="alert alert-danger">{errorMessage}</div>}
      <div className="list-group">
        {contents.directories.map((dir) => (
          <button
            key={dir}
            className="list-group-item list-group-item-action"
            onClick={() => handleDirectoryClick(dir)}
          >
            {dir}
          </button>
        ))}
        {contents.files.map((file) => (
          <button
            key={file}
            className="list-group-item list-group-item-action"
            onClick={() => handleDownloadFile(file.url)}
          >
            {file.name}
          </button>
        ))}
      </div>
      {contents.directories.length === 0 && contents.files.length === 0 && (
        <div className="alert alert-info mt-3">No files or directories found.</div>
      )}
    </div>
  );
};

export default UserFilesForm;
