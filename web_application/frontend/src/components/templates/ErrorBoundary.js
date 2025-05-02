import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExclamationTriangle } from '@fortawesome/free-solid-svg-icons';
import styled from '@emotion/styled';

const ErrorContainer = styled.div`
  margin: 2rem;
  padding: 2rem;
  background-color: #fff3f3;
  border-left: 5px solid #ff5252;
  border-radius: 4px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const ErrorHeading = styled.h2`
  color: #d32f2f;
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin-top: 0;
`;

const ErrorMessage = styled.div`
  margin: 1rem 0;
  padding: 1rem;
  background-color: #f8f8f8;
  border-radius: 4px;
  font-family: monospace;
  white-space: pre-wrap;
  overflow: auto;
  max-height: 300px;
`;

const RetryButton = styled.button`
  background-color: #d32f2f;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
  transition: background-color 0.2s;

  &:hover {
    background-color: #b71c1c;
  }
`;

/**
 * Error boundary component to handle JavaScript errors
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log error information
    console.error('ErrorBoundary caught an error', error, errorInfo);
    this.setState({ errorInfo });

    // You could also log to an error reporting service here
  }

  handleRetry = () => {
    // Reset the error state and allow component to re-render
    this.setState({ hasError: false, error: null, errorInfo: null });

    // If there's a custom retry handler, call it
    if (this.props.onRetry) {
      this.props.onRetry();
    }
  };

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      return (
        <ErrorContainer role="alert">
          <ErrorHeading>
            <FontAwesomeIcon icon={faExclamationTriangle} />
            Something went wrong
          </ErrorHeading>
          <p>
            There was an error loading this component. You can try reloading the page, or check the
            console for more details.
          </p>
          {this.state.error && <ErrorMessage>{this.state.error.toString()}</ErrorMessage>}
          <RetryButton onClick={this.handleRetry}>Try Again</RetryButton>
        </ErrorContainer>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
