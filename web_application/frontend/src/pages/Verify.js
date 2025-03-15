import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import VerificationForm from '../components/forms/Verification';

const Verify = () => {
  const [verificationCode, setVerificationCode] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const navigate = useNavigate();

  const [email, setEmail] = useState('');

  // Add useEffect to check for stored email from login redirect
  useEffect(() => {
    const storedEmail = localStorage.getItem('verificationEmail');
    if (storedEmail) {
      setEmail(storedEmail);
      // Clear it from localStorage after using it
      localStorage.removeItem('verificationEmail');
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');

    try {
      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, verification_code: verificationCode }),
      });

      const result = await response.json();

      if (response.ok) {
        setSuccessMessage(result.message || 'Verification successful!');
        setTimeout(() => {
          navigate('/login');
        }, 1000);
      } else {
        setErrorMessage(result.message || 'Verification failed. Please try again.');
      }
    } catch (error) {
      setErrorMessage('An error occurred. Please try again.');
    }
  };

  return (
    <div className="container my-5">
      <div className="row justify-content-center">
        <div className="col-md-6 col-lg-5">
          <div className="card shadow-sm">
            <div className="card-body">
              <h2 className="text-center mb-4">Email Verification</h2>
              <p className="text-center">
                Please enter the verification code we sent to your email.
              </p>

              {errorMessage && <div className="alert alert-danger">{errorMessage}</div>}
              {successMessage && <div className="alert alert-success">{successMessage}</div>}

              <VerificationForm
                verificationCode={verificationCode}
                setVerificationCode={setVerificationCode}
                email={email}
                setEmail={setEmail}
                handleSubmit={handleSubmit}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Verify;
