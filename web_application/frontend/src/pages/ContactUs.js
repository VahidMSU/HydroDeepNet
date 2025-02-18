import React, { useState } from 'react';
import axios from 'axios';
import ContactUsTemplate from '../components/templates/ContactUs';

const ContactUs = () => {
  const [flashMessages, setFlashMessages] = useState([]);

  const handleFormSubmit = async (formData) => {
    try {
      const response = await axios.post('/contact', formData, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      setFlashMessages([{ category: 'success', text: response.data.message }]);
    } catch (error) {
      const errorMsg =
        error.response?.data?.message || 'Failed to submit the message. Please try again later.';
      setFlashMessages([{ category: 'error', text: errorMsg }]);
    }
  };

  return (
    <ContactUsTemplate
      handleFormSubmit={handleFormSubmit}
      flashMessages={flashMessages}
      setFlashMessages={setFlashMessages}
    />
  );
};

export default ContactUs;
