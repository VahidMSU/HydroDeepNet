// pages/UserDashboard.js
import React, { useEffect, useState } from 'react';
import '../css/UserDashboard.css'; // Adjust the path if necessary

const UserDashboard = () => {
  const [files, setFiles] = useState([]);

  useEffect(() => {
    // TODO: Replace with actual logic/API call to fetch file list
    const fetchFiles = async () => {
      // Example: Fetch file list from an API endpoint
      // const response = await fetch("/api/user/files");
      // const data = await response.json();
      // setFiles(data);

      // For demonstration purposes, using static data:
      setFiles(['file1.txt', 'file2.txt', 'file3.txt']);
    };

    fetchFiles();
  }, []);

  return (
    <main className="container my-5">
      <h1 className="text-center mb-4">User Dashboard</h1>
      <section id="user-data">
        <h2>Your Data Directory</h2>
        <div id="file-list" className="file-list">
          {files.length ? (
            files.map((file, index) => (
              <div key={index} className="file-item">
                {file}
              </div>
            ))
          ) : (
            <p>No files found.</p>
          )}
        </div>
      </section>
    </main>
  );
};

export default UserDashboard;
