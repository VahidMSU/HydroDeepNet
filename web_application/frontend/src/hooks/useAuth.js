import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // Logic to check if the user is authenticated
    const token = localStorage.getItem('authToken');
    setIsAuthenticated(!!token);

    if (!token) {
      navigate('/login');
    }
  }, [navigate]);

  return { isAuthenticated };
};

export default useAuth;
