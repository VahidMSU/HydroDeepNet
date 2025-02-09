// pages/Verify.js
import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import VerificationForm from '../components/forms/Verification'; // Import the new component

const Verify = () => {
  const [verificationCode, setVerificationCode] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // TODO: Add your verification logic here, e.g., an API call.
    console.log('Verification code submitted:', verificationCode);
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
              <VerificationForm
                verificationCode={verificationCode}
                setVerificationCode={setVerificationCode}
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
