import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faInfoCircle, faTimes } from '@fortawesome/free-solid-svg-icons';
import styled from '@emotion/styled';
import colors from '../styles/colors.tsx';

const InfoBoxContainer = styled.div`
  background-color: ${colors.surface};
  border-left: 4px solid ${colors.info};
  border-radius: 8px;
  padding: 1.2rem;
  margin-bottom: 1.5rem;
  position: relative;
  animation: fadeIn 0.3s ease-in-out;

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
`;

const InfoHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.8rem;

  h3 {
    color: ${colors.info};
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
    font-size: 1.1rem;

    .icon {
      color: ${colors.info};
    }
  }
`;

const CloseButton = styled.button`
  background: transparent;
  border: none;
  color: ${colors.textMuted};
  cursor: pointer;
  padding: 0.3rem;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color 0.2s;

  &:hover {
    color: ${colors.text};
  }
`;

const InfoContent = styled.div`
  color: ${colors.textSecondary};
  font-size: 0.95rem;
  line-height: 1.5;

  ul {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
  }

  li {
    margin-bottom: 0.5rem;
  }
`;

const InfoBox = ({ title, children, dismissible = true }) => {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  return (
    <InfoBoxContainer>
      <InfoHeader>
        <h3>
          <FontAwesomeIcon icon={faInfoCircle} className="icon" />
          {title}
        </h3>
        {dismissible && (
          <CloseButton onClick={() => setDismissed(true)}>
            <FontAwesomeIcon icon={faTimes} />
          </CloseButton>
        )}
      </InfoHeader>
      <InfoContent>{children}</InfoContent>
    </InfoBoxContainer>
  );
};

export default InfoBox;
