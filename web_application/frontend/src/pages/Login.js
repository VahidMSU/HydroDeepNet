import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

const Login = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    remember_me: false,
  });
  const [errors, setErrors] = useState({});

  const handleChange = ({ target: { name, value, type, checked } }) => {
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrors({});

    const newErrors = {};
    if (!formData.username) {
      newErrors.username = 'Username is required';
    }
    if (!formData.password) {
      newErrors.password = 'Password is required';
    }
    if (Object.keys(newErrors).length) {
      setErrors(newErrors);
      return;
    }

    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      if (data.success) {
        // store token or other credentials as needed
        localStorage.setItem('authToken', data.token);
        navigate('/');
      } else {
        setErrors({ login: data.message });
      }
    } catch (error) {
      console.error('Login error:', error);
      setErrors({ login: 'Login failed. Try again.' });
    }
  };

  return (
    <div className="container my-5">
      <div className="row justify-content-center">
        <div className="col-md-6 col-lg-4">
          <div className="card shadow">
            <div className="card-body p-4">
              <h2 className="text-center mb-4">Login</h2>

              {/* MSU NetID Login Button */}
              <div className="d-grid mb-3">
                <a href="/login?msu_oauth=True" className="btn btn-success">
                  Login with MSU NetID
                </a>
              </div>

              {/* Standard Login Form */}
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
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
