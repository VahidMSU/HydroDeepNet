import React, { useState } from 'react';
import SignUpTemplate from '../components/templates/SignUp'; // Import the new SignUpTemplate component

const SignUp = () => {
  const [flashMessages, setFlashMessages] = useState([]);

  const handleFormSubmit = async (formData) => {
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
      } else {
        setFlashMessages([{ category: 'error', text: result.message }]);
      }
    } catch (error) {
      setFlashMessages([
        { category: 'error', text: 'An error occurred during signup. Please try again.' },
      ]);
    }
  };

  return <SignUpTemplate flashMessages={flashMessages} handleFormSubmit={handleFormSubmit} />;
};

export default SignUp;
