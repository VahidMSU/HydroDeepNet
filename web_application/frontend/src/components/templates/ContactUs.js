import React from 'react';
import ContactUsForm from '../forms/ContactUs.js';

const ContactUsTemplate = ({ handleFormSubmit, flashMessages, setFlashMessages }) => {
  return <ContactUsForm onSubmit={handleFormSubmit} />;
};

export default ContactUsTemplate;
