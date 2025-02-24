import React from 'react';
import ContactUsForm from '../forms/ContactUs.js';
import { ContactContainer, ContactTitle, ContentWrapper, Alert } from '../../styles/ContactUs.tsx';

const ContactUsTemplate = ({ handleFormSubmit, flashMessages, setFlashMessages }) => {
  return (
    <ContactUsForm onSubmit={handleFormSubmit} />
  );
};

export default ContactUsTemplate;
