import React from 'react';
import { Link } from 'react-router-dom';
import SignUpForm from '../forms/SignUp.js';
import '../../styles/SignUp.tsx'; // Corrected CSS file path

const SignUpTemplate = ({ flashMessages, handleFormSubmit }) => {
  return (
    <div className="signup-page">
      <div className="signup-container">
        <h2>Sign Up</h2>

        {flashMessages.length > 0 && (
          <div className="alert alert-info">
            {flashMessages.map((msg, idx) => (
              <div key={idx} className={`alert alert-${msg.category}`}>
                {msg.text}
              </div>
            ))}
          </div>
        )}

        <SignUpForm onSubmit={handleFormSubmit} />

        <footer>
          By signing up, you agree to our <Link to="/privacy">Privacy Policy</Link> and{' '}
          <Link to="/terms">Terms of Service</Link>.
        </footer>
      </div>
    </div>
  );
};

export default SignUpTemplate;
