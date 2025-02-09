import React from 'react';

const VerificationForm = ({ verificationCode, setVerificationCode, handleSubmit }) => (
  <form onSubmit={handleSubmit}>
    <div className="mb-3">
      <label htmlFor="verificationCode" className="form-label">
        Verification Code
      </label>
      <input
        type="text"
        className="form-control"
        id="verificationCode"
        placeholder="Enter verification code"
        value={verificationCode}
        onChange={(e) => setVerificationCode(e.target.value)}
      />
    </div>
    <div className="d-grid">
      <button type="submit" className="btn btn-primary">
        Submit
      </button>
    </div>
  </form>
);

export default VerificationForm;
