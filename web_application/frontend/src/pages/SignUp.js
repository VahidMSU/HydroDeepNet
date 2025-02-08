import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import '../css/SignUp.css'; // Corrected CSS file path

const SignUp = () => {
  // Simulated flash messages (in a real app, these might come from props or context)
  const [flashMessages, setFlashMessages] = useState([]);

  // Form state for controlled components
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirm_password: '',
  });

  // Validation errors
  const [errors, setErrors] = useState({});

  // Update form fields as the user types
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // Basic client-side validation
  const validate = () => {
    const newErrors = {};
    if (!formData.username) {
      newErrors.username = 'Username is required';
    }
    if (!formData.email) {
      newErrors.email = 'Email is required';
    }
    if (!formData.password) {
      newErrors.password = 'Password is required';
    }
    if (formData.password !== formData.confirm_password) {
      newErrors.confirm_password = 'Passwords must match';
    }
    return newErrors;
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    const validationErrors = validate();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    try {
      const response = await fetch('/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      if (response.ok) {
        setFlashMessages([{ category: 'info', text: result.message }]);
        setErrors({});
        setFormData({
          username: '',
          email: '',
          password: '',
          confirm_password: '',
        });
      } else {
        setFlashMessages([{ category: 'error', text: result.message }]);
        setErrors(result.errors || {});
      }
    } catch (error) {
      setFlashMessages([
        { category: 'error', text: 'An error occurred during signup. Please try again.' },
      ]);
    }
  };

  return (
    <div className="signup-page">
      <div className="signup-container">
        <h2>Sign Up</h2>

        {/* Flash Messages */}
        {flashMessages.length > 0 && (
          <div className="alert alert-info">
            {flashMessages.map((msg, idx) => (
              <div key={idx} className={`alert alert-${msg.category}`}>
                {msg.text}
              </div>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {/* Username Field */}
          <div className="form-group">
            <label htmlFor="username">Username:</label>
            <input
              type="text"
              className="form-control"
              name="username"
              id="username"
              value={formData.username}
              onChange={handleChange}
            />
            {errors.username && <div className="error-message">{errors.username}</div>}
          </div>

          {/* Email Field */}
          <div className="form-group">
            <label htmlFor="email">Email:</label>
            <input
              type="email"
              className="form-control"
              name="email"
              id="email"
              value={formData.email}
              onChange={handleChange}
            />
            {errors.email && <div className="error-message">{errors.email}</div>}
          </div>

          {/* Password Field */}
          <div className="form-group">
            <label htmlFor="password">Password:</label>
            <input
              type="password"
              className="form-control"
              name="password"
              id="password"
              value={formData.password}
              onChange={handleChange}
            />
            {errors.password && <div className="error-message">{errors.password}</div>}
          </div>

          {/* Confirm Password Field */}
          <div className="form-group">
            <label htmlFor="confirm_password">Confirm Password:</label>
            <input
              type="password"
              className="form-control"
              name="confirm_password"
              id="confirm_password"
              value={formData.confirm_password}
              onChange={handleChange}
            />
            {errors.confirm_password && (
              <div className="error-message">{errors.confirm_password}</div>
            )}
          </div>

          {/* Submit Button */}
          <button type="submit" className="btn btn-primary">
            Sign Up
          </button>
        </form>

        {/* General Form Error */}
        {Object.keys(errors).length > 0 && (
          <div className="alert alert-danger mt-3">Please fix the errors above and try again.</div>
        )}
      </div>

      {/* Footer Section */}
      <footer>
        By signing up, you agree to our <Link to="/privacy">Privacy Policy</Link> and{' '}
        <Link to="/terms">Terms of Service</Link>.
      </footer>
    </div>
  );
};

export default SignUp;
