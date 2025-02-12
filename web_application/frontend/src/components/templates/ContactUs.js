import React from 'react';
import ContactUsForm from '../forms/ContactUs.js';
import {
  ContactUsContainer,
  ContactHeader,
  ContactCard,
  CardBody,
  Alert,
} from '../../styles/ContactUs.tsx';

const ContactUsTemplate = ({ handleFormSubmit, flashMessages, setFlashMessages }) => {
  return (
    <ContactUsContainer>
      <ContactHeader>
        <h2>Contact Us</h2>
        <p>
          Have questions, feedback, or collaboration ideas? We'd love to hear from you! Please fill
          out the form below, and we'll get back to you as soon as possible.
        </p>
      </ContactHeader>

      <ContactCard>
        <CardBody>
          {flashMessages.length > 0 &&
            flashMessages.map((msg, idx) => (
              <Alert key={idx} className={`alert-${msg.category}`}>
                {msg.text}
                <button type="button" onClick={() => setFlashMessages([])}>
                  ×
                </button>
              </Alert>
            ))}

          <ContactUsForm onSubmit={handleFormSubmit} />
        </CardBody>
      </ContactCard>
    </ContactUsContainer>
  );
};

export default ContactUsTemplate;
