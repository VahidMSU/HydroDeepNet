document.addEventListener("DOMContentLoaded", function () {
  // Define the function BEFORE calling it
  const loadUserFiles = function (subdir) {
    const endpoint = `${
      window.location.origin
    }/api/user_files?subdir=${encodeURIComponent(subdir)}`;

    fetch(endpoint)
      .then((response) => response.json())
      .then((data) => {
        const fileListContainer = document.getElementById("file-list");
        fileListContainer.innerHTML = ""; // Clear previous content

        // Navigation UI (Back to Parent Directory)
        if (data.parent_path !== "") {
          const parentItem = document.createElement("div");
          parentItem.classList.add("file-item", "directory");
          parentItem.innerHTML = "⬆️ .. (Parent Directory)";
          parentItem.addEventListener("click", () =>
            loadUserFiles(data.parent_path)
          );
          fileListContainer.appendChild(parentItem);
        }

        // Display Directories
        data.directories.forEach((dir) => {
          const dirItem = document.createElement("div");
          dirItem.classList.add("file-item", "directory");

          const dirName = document.createElement("span");
          dirName.textContent = `📁 ${dir.name}`;
          dirName.classList.add("file-name");
          dirName.addEventListener("click", () => loadUserFiles(dir.path));

          const downloadDirBtn = document.createElement("a");
          downloadDirBtn.classList.add("download-btn");
          downloadDirBtn.textContent = "Download ZIP";
          downloadDirBtn.href = dir.download_zip_url;
          downloadDirBtn.setAttribute("download", `${dir.name}.zip`);

          dirItem.appendChild(dirName);
          dirItem.appendChild(downloadDirBtn);
          fileListContainer.appendChild(dirItem);
        });

        // Display Files
        data.files.forEach((file) => {
          const fileItem = document.createElement("div");
          fileItem.classList.add("file-item");

          const fileName = document.createElement("span");
          fileName.classList.add("file-name");
          fileName.textContent = `📄 ${file.name}`;

          const downloadBtn = document.createElement("a");
          downloadBtn.classList.add("download-btn");
          downloadBtn.textContent = "Download";
          downloadBtn.href = file.download_url;
          downloadBtn.setAttribute("download", file.name);

          fileItem.appendChild(fileName);
          fileItem.appendChild(downloadBtn);
          fileListContainer.appendChild(fileItem);
        });

        // Handle empty directory case
        if (data.directories.length === 0 && data.files.length === 0) {
          fileListContainer.innerHTML = "<p>No files or folders found.</p>";
        }
      })
      .catch((error) => {
        console.error("Error fetching user files:", error);
        document.getElementById("file-list").innerHTML =
          "<p>Error loading files.</p>";
      });
  };

  // Now, it's safe to call the function
  loadUserFiles(""); // Load root directory on page load
});
