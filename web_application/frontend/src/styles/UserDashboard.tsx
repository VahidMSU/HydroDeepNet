/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';
import colors from './colors.tsx';

export const DashboardContainer = styled.div`
  padding: 2rem;
  background-color: ${colors.background};
  min-height: 100vh;
  color: ${colors.text};
`;

export const DashboardHeader = styled.div`
  text-align: center;
  margin-bottom: 2.5rem;
  
  h1 {
    color: ${colors.accent};
    font-size: 2.5rem;
    margin-bottom: 1rem;
    border-bottom: 3px solid ${colors.accent};
    padding-bottom: 1rem;
    position: relative;
    
    &:after {
      content: '';
      position: absolute;
      bottom: -3px;
      left: 50%;
      transform: translateX(-50%);
      width: 100px;
      height: 3px;
      background: linear-gradient(to right, transparent, ${colors.accentHover}, transparent);
    }
  }
`;

export const ContentGrid = styled.div`
  display: grid;
  gap: 1.8rem;
  margin: 2rem 0;
  
  /* Different view modes */
  &.grid-large {
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  }
  
  &.grid-small {
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  }
  
  &.list-view {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
  }
`;

// Base card with shared properties
const BaseCard = styled.div`
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  transition: all 0.25s ease-in-out;
  height: 100%;
  display: flex;
  flex-direction: column;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  }
  
  /* List view styling */
  .list-view & {
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    height: auto;
    
    .item-details {
      display: flex;
      align-items: center;
      flex: 1;
    }
    
    .item-metadata {
      display: flex;
      gap: 2rem;
      align-items: center;
      margin-right: 1.5rem;
    }
    
    .item-actions {
      display: flex;
      gap: 0.5rem;
    }
  }
  
  /* Small grid view styling */
  .grid-small & {
    padding: 1rem;
    
    h3 {
      font-size: 0.9rem;
    }
    
    .icon {
      font-size: 1.5rem;
    }
  }
`;

// Folder card with distinctive styling
export const FolderCard = styled(BaseCard)`
  background: linear-gradient(135deg, ${colors.surfaceLight} 0%, ${colors.surface} 100%);
  border-left: 4px solid ${colors.accent};
  
  &:hover {
    border-left-color: ${colors.accentHover};
  }
  
  .list-view & {
    border-left: none;
    border-left: 4px solid ${colors.accent};
  }
`;

// File card with distinctive styling
export const FileCard = styled(BaseCard)`
  background: linear-gradient(135deg, ${colors.surfaceDark} 0%, ${colors.surface} 100%);
  border-left: 4px solid ${colors.info};
  
  &:hover {
    border-left-color: #52b0ff; /* Lighter info color */
  }
  
  .list-view & {
    border-left: none;
    border-left: 4px solid ${colors.info};
  }
`;

// Base header for both file and folder cards
const BaseHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin-bottom: 1rem;
  padding-bottom: 0.8rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);

  h3 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .list-view & {
    margin-bottom: 0;
    padding-bottom: 0;
    border-bottom: none;
    flex: 0 0 30%;
  }
  
  .grid-small & {
    padding-bottom: 0.5rem;
    margin-bottom: 0.5rem;
    
    h3 {
      font-size: 0.9rem;
    }
  }
`;

// Folder header with distinctive styling
export const FolderHeader = styled(BaseHeader)`
  .icon {
    color: ${colors.accent};
    font-size: 1.8rem;
  }

  h3 {
    color: ${colors.text};
  }
`;

// File header with distinctive styling
export const FileHeader = styled(BaseHeader)`
  .icon {
    color: ${colors.info};
    font-size: 1.8rem;
  }

  h3 {
    color: #ffffff;
  }
`;

export const ItemInfo = styled.div`
  margin: 0.8rem 0;
  font-size: 0.9rem;
  color: ${colors.textSecondary};

  p {
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .info-icon {
    color: ${colors.textMuted};
    font-size: 0.9rem;
  }
  
  .list-view & {
    margin: 0;
    display: flex;
    align-items: center;
    gap: 2rem;
    
    p {
      margin: 0;
    }
  }
  
  .grid-small & {
    margin: 0.4rem 0;
    font-size: 0.8rem;
    
    p {
      margin: 0.2rem 0;
    }
  }
`;

const BaseButton = styled.button`
  border: none;
  border-radius: 8px;
  padding: 0.9rem 1rem;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  transition: all 0.2s ease-in-out;
  margin-top: auto; /* Push button to bottom of card */
  text-transform: uppercase;
  letter-spacing: 0.5px;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }

  &:active {
    transform: translateY(0);
  }

  .icon {
    font-size: 1.2rem;
  }
  
  .list-view & {
    margin-top: 0;
    width: auto;
    padding: 0.5rem 0.8rem;
    font-size: 0.85rem;
  }
  
  .grid-small & {
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
  }
`;

// Folder button styling
export const FolderButton = styled(BaseButton)`
  background-color: ${colors.accent};
  color: ${colors.textInverse};
  
  &:hover {
    background-color: ${colors.accentHover};
  }
`;

// File button styling 
export const FileButton = styled(BaseButton)`
  background-color: ${colors.info};
  color: white;
  
  &:hover {
    background-color: #52b0ff; /* Lighter info color */
  }
  
  &.icon-only {
    width: 45px;
    height: 45px;
    padding: 0;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    
    .icon {
      margin: 0;
      font-size: 1.2rem;
    }
  }
`;

export const BreadcrumbNav = styled.div`
  margin-bottom: 2.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  background-color: ${colors.surfaceDark};
  padding: 1rem 1.5rem;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);

  .home-icon {
    color: ${colors.accent};
    font-size: 1.4rem;
  }

  .separator {
    color: ${colors.textMuted};
    margin: 0 0.3rem;
  }

  button {
    background: none;
    border: none;
    color: ${colors.accent};
    cursor: pointer;
    font-size: 1rem;
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;

    &:hover {
      background-color: rgba(255, 133, 0, 0.15);
      transform: translateY(-2px);
    }
    
    &:active {
      transform: translateY(0);
    }

    &.current {
      color: ${colors.text};
      cursor: default;
      
      &:hover {
        transform: none;
        background-color: transparent;
      }
    }
  }
`;

// View mode control styles
export const ViewControls = styled.div`
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.8rem;
  margin-bottom: 1.5rem;
  background-color: ${colors.surfaceDark};
  padding: 0.8rem 1rem;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
`;

export const ViewButton = styled.button`
  background: none;
  border: none;
  color: ${colors.textSecondary};
  cursor: pointer;
  font-size: 1.2rem;
  width: 40px;
  height: 40px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  
  &:hover {
    color: ${colors.accent};
    background-color: rgba(255, 133, 0, 0.15);
  }
  
  &.active {
    color: ${colors.accent};
    background-color: rgba(255, 133, 0, 0.25);
  }
`;

export const ViewModeLabel = styled.span`
  color: ${colors.textSecondary};
  margin-right: 1rem;
  font-size: 0.9rem;
`;

export const ListViewHeader = styled.div`
  display: grid;
  grid-template-columns: 30% 1fr 15%;
  padding: 0.8rem 1.5rem;
  border-radius: 10px;
  background-color: ${colors.surfaceDark};
  color: ${colors.textSecondary};
  font-weight: 600;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
  
  .name {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .details {
    display: flex;
    gap: 2rem;
  }
  
  .actions {
    text-align: right;
  }
`;

export const EmptyState = styled.div`
  text-align: center;
  padding: 4rem 2rem;
  color: ${colors.textSecondary};
  background-color: ${colors.surfaceDark};
  border-radius: 12px;
  border: 1px dashed ${colors.border};
  grid-column: 1 / -1;

  .icon {
    font-size: 4rem;
    color: ${colors.accent};
    margin-bottom: 1.5rem;
    opacity: 0.8;
  }

  h3 {
    margin-bottom: 1rem;
    font-size: 1.5rem;
    color: ${colors.text};
  }
  
  p {
    font-size: 1.1rem;
  }
`;

export const ErrorMessage = styled.div`
  background-color: rgba(244, 67, 54, 0.1);
  border: 1px solid ${colors.error};
  color: ${colors.error};
  padding: 1.2rem;
  border-radius: 8px;
  margin-bottom: 2rem;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  
  .error-icon {
    font-size: 1.5rem;
  }
`;

export const FileTypeIcon = styled.div`
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  font-weight: bold;
  margin-right: 0.8rem;
  text-transform: uppercase;
  flex-shrink: 0;
  
  &.pdf {
    background-color: #f44336;
    color: white;
  }
  
  &.txt {
    background-color: #4caf50;
    color: white;
  }
  
  &.csv, &.xlsx {
    background-color: #2196f3;
    color: white;
  }
  
  &.zip {
    background-color: #ff9800;
    color: white;
  }
  
  &.img {
    background-color: #9c27b0;
    color: white;
  }
  
  &.default {
    background-color: #757575;
    color: white;
  }
`;

export const BadgeCount = styled.div`
  background-color: ${colors.accent};
  color: ${colors.textInverse};
  border-radius: 12px;
  padding: 0.3rem 0.6rem;
  font-size: 0.8rem;
  font-weight: bold;
  margin-left: 0.5rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
`;

export const DirectoryGuide = styled.div`
  margin: 1.5rem 0;
  padding: 1.5rem;
  background-color: ${colors.surfaceDark};
  border-radius: 10px;
  border: 1px solid ${colors.border};
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  
  h3 {
    margin-top: 0;
    color: ${colors.accent};
    display: flex;
    align-items: center;
    font-size: 1.3rem;
    margin-bottom: 1rem;
    
    .guide-icon {
      margin-right: 0.8rem;
      color: ${colors.accent};
    }
  }
  
  p {
    margin-bottom: 1rem;
    color: ${colors.text};
  }
  
  ul {
    padding-left: 1.5rem;
    margin-bottom: 1rem;
    color: ${colors.text};
    
    li {
      margin-bottom: 0.5rem;
      
      strong {
        color: ${colors.accent};
      }
      
      code {
        background-color: ${colors.surface};
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: monospace;
        color: ${colors.accent};
      }
    }
  }
  
  .guide-footer {
    margin: 0;
    font-style: italic;
    color: ${colors.textSecondary};
  }
`;
