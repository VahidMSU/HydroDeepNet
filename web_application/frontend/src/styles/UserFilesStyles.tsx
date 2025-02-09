/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';

const FileContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  width: 100%;
`;

const DirectoryContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  width: 100%;
  margin-bottom: 20px;
`;

const FileList = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  width: 100%;
`;

const DirectoryItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin-bottom: 10px;
`;

const FileItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin-bottom: 10px;
`;

const DirectoryButton = styled.button`
  background-color: #007bff;
  color: #ffffff;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  &:hover {
    background-color: #0056b3;
  }
`;

const DownloadButton = styled.button`
  background-color: #28a745;
  color: #ffffff;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  &:hover {
    background-color: #218838;
  }
`;

const ErrorMessage = styled.p`
  color: #dc3545;
  margin-bottom: 20px;
`;

const GoUpButton = styled.button`
  background-color: #ffc107;
  color: #ffffff;
  border: none;
  padding: 10px 20px;
  margin: 5px 0;
  border-radius: 4px;
  cursor: pointer;
  &:hover {
    background-color: #e0a800;
  }
`;

export { FileContainer, DirectoryButton, DownloadButton, ErrorMessage, GoUpButton, DirectoryContainer, FileList, DirectoryItem, FileItem };
