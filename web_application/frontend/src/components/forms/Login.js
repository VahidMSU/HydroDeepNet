import React from 'react';
import { Link } from 'react-router-dom';

const LoginForm = ({ formData, handleChange, handleSubmit, errors }) => {
  return (
    <form onSubmit={handleSubmit}>
      {/* Username Field */}
      <div className="mb-3">
        <label htmlFor="username" className="form-label">
          Username:
        </label>
        <input
          type="text"
          id="username"
          name="username"
          className={`form-control ${errors.username ? 'is-invalid' : ''}`}
          placeholder="Enter username"
          autoFocus
          value={formData.username}
          onChange={handleChange}
        />
        {errors.username && <div className="invalid-feedback">{errors.username}</div>}
      </div>

      {/* Password Field */}
      <div className="mb-3">
        <label htmlFor="password" className="form-label">
          Password:
        </label>
        <input
          type="password"
          id="password"
          name="password"
          className={`form-control ${errors.password ? 'is-invalid' : ''}`}
          placeholder="Enter password"
          value={formData.password}
          onChange={handleChange}
        />
        {errors.password && <div className="invalid-feedback">{errors.password}</div>}
      </div>

      {/* Remember Me Checkbox */}
      <div className="mb-3 form-check">
        <input
          type="checkbox"
          id="remember_me"
          name="remember_me"
          className="form-check-input"
          checked={formData.remember_me}
          onChange={handleChange}
        />
        <label className="form-check-label" htmlFor="remember_me">
          Remember Me
        </label>
      </div>

      {/* Submit Button */}
      <div className="d-grid">
        <button type="submit" className="btn btn-primary">
          Login
        </button>
      </div>

      {/* Additional Links */}
      <div className="text-center mt-3">
        <p>
          Don&apos;t have an account? <Link to="/signup">Sign up</Link>
        </p>

        <p>
          <Link to="/">Return to Home</Link>
        </p>
      </div>
    </form>
  );
};

export default LoginForm;
