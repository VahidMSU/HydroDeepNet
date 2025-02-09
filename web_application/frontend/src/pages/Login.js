import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import LoginTemplate from '../components/templates/Login'; // Import the new LoginTemplate component

const Login = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    remember_me: false,
  });
  const [errors, setErrors] = useState({});

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (token) {
      navigate('/');
    }
  }, [navigate]);

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
        alert(data.message); // Flash message for incorrect username or password
      }
    } catch (error) {
      console.error('Login error:', error);
      setErrors({ login: 'Login failed. Try again.' });
    }
  };

  return (
    <LoginTemplate
      formData={formData}
      handleChange={handleChange}
      handleSubmit={handleSubmit}
      errors={errors}
    />
  );
};

export default Login;
