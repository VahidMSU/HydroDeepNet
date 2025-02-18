import React from 'react';
import ContactUsForm from '../forms/ContactUs.js';
import { ContactContainer, ContactTitle, ContentWrapper, Alert } from '../../styles/ContactUs.tsx';

const ContactUsTemplate = ({ handleFormSubmit, flashMessages, setFlashMessages }) => {
  return (
    <ContactContainer>
      <ContactTitle>Contact Us</ContactTitle>

      <ContentWrapper>
        {flashMessages.length > 0 &&
          flashMessages.map((msg, idx) => (
            <Alert key={idx} type={msg.category}>
              {msg.text}
              <button type="button" onClick={() => setFlashMessages([])}>
                ×
              </button>
            </Alert>
          ))}

        <ContactUsForm onSubmit={handleFormSubmit} />
      </ContentWrapper>
    </ContactContainer>
  );
};

export default ContactUsTemplate;
