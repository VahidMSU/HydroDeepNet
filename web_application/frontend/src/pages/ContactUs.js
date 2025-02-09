import React, { useState } from 'react';
import ContactUsTemplate from '../components/templates/ContactUs'; // Import the new ContactUsTemplate component

const ContactUs = () => {
  const [flashMessages, setFlashMessages] = useState([]);

  const handleFormSubmit = (formData) => {
    console.log('Submitted:', formData);
    setFlashMessages([{ category: 'success', text: 'Your message has been sent!' }]);
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
