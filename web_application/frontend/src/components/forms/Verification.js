///data/SWATGenXApp/codes/web_application/frontend/src/components/forms/Verification.js
import React from 'react';

const VerificationForm = ({
  verificationCode,
  setVerificationCode,
  email,
  setEmail,
  handleSubmit,
}) => {
  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group mb-3">
        <label htmlFor="email">Email</label>
        <input
          type="email"
          id="email"
          className="form-control"
          placeholder="Enter your email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </div>

      <div className="form-group mb-3">
        <label htmlFor="verificationCode">Verification Code</label>
        <input
          type="text"
          id="verificationCode"
          className="form-control"
          value={verificationCode}
          onChange={(e) => setVerificationCode(e.target.value)}
          required
        />
      </div>

      <button type="submit" className="btn btn-primary w-100">
        Verify
      </button>
    </form>
  );
};

export default VerificationForm;
