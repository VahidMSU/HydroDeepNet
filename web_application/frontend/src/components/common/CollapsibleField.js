import React, { useState } from 'react';
import styled from '@emotion/styled';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronDown, faChevronUp } from '@fortawesome/free-solid-svg-icons';

const CollapsibleContainer = styled.div`
  margin-bottom: 0.75rem;
  border: 1px solid #687891;
  border-radius: 6px;
  overflow: hidden;
`;

const CollapsibleHeader = styled.div`
  padding: 8px 12px;
  background-color: #444e5e;
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background-color 0.2s ease;

  &:hover {
    background-color: #505b6b;
  }
`;

const CollapsibleContent = styled.div`
  padding: ${(props) => (props.isOpen ? '12px' : '0')};
  max-height: ${(props) => (props.isOpen ? '800px' : '0')};
  opacity: ${(props) => (props.isOpen ? '1' : '0')};
  transition: all 0.3s ease-in-out;
  background-color: #ffffff;
`;

const CollapsibleField = ({ title, children, defaultOpen = false }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <CollapsibleContainer>
      <CollapsibleHeader onClick={() => setIsOpen(!isOpen)}>
        <span>{title}</span>
        <FontAwesomeIcon icon={isOpen ? faChevronUp : faChevronDown} />
      </CollapsibleHeader>
      <CollapsibleContent isOpen={isOpen}>{children}</CollapsibleContent>
    </CollapsibleContainer>
  );
};

export default CollapsibleField;
