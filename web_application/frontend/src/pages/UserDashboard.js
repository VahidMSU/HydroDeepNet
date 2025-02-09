import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../styles/UserDashboard.tsx'; // Adjust the path if necessary

const UserDashboard = () => {
  const [contents, setContents] = useState({ directories: [], files: [] });
  const [currentPath, setCurrentPath] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const fetchFiles = async (path = '') => {
    try {
      const response = await axios.get('/api/user_files', { params: { subdir: path } });
      setContents(response.data);
      setCurrentPath(path);
    } catch (error) {
      console.error('Error fetching files:', error);
      setErrorMessage('Failed to fetch files.');
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const handleDirectoryClick = (path) => {
    fetchFiles(path);
  };

  const handleDownloadFile = (url) => {
    window.location.href = url;
  };

  const handleDownloadDirectory = (url) => {
    window.location.href = url;
  };

  return (
    <main className="container my-5">
      <h1 className="text-center mb-4">User Dashboard</h1>
      <section id="user-data">
        <h2>Your Data Directory</h2>
        {errorMessage && (
          <div className="alert alert-danger" role="alert">
            {errorMessage}
          </div>
        )}
        <div id="file-list" className="file-list">
          {contents.directories.length || contents.files.length ? (
            <>
              {contents.parent_path && (
                <div
                  className="file-item"
                  onClick={() => handleDirectoryClick(contents.parent_path)}
                >
                  <strong>.. (Parent Directory)</strong>
                </div>
              )}
              {contents.directories.map((dir, index) => (
                <div
                  key={index}
                  className="file-item"
                  onClick={() => handleDirectoryClick(dir.path)}
                >
                  <strong>Directory:</strong> {dir.name}
                  <button onClick={() => handleDownloadDirectory(dir.download_zip_url)}>
                    Download ZIP
                  </button>
                </div>
              ))}
              {contents.files.map((file, index) => (
                <div key={index} className="file-item">
                  <strong>File:</strong> {file.name}
                  <button onClick={() => handleDownloadFile(file.download_url)}>Download</button>
                </div>
              ))}
            </>
          ) : (
            <p>No files or directories found.</p>
          )}
        </div>
      </section>
    </main>
  );
};

export default UserDashboard;
