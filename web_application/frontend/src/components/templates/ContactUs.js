import React, { useState } from 'react';
import ContactUsForm from '../forms/ContactUs.js';
import {
  ContactUsContainer,
  Card as ContactUsCard,
  CardBody as ContactUsCardBody,
  TextCenter as ContactUsIntroText,
} from '../../styles/ContactUs.tsx'; // Import styled components

const ContactUsTemplate = ({ handleFormSubmit, flashMessages, setFlashMessages }) => {
  return (
    <ContactUsContainer>
      <ContactUsIntroText>
        <h2>Contact Us</h2>
      </ContactUsIntroText>
      <ContactUsCard>
        <ContactUsCardBody>
          <ContactUsIntroText>
            <p>
              Have questions, feedback, or collaboration ideas? We'd love to hear from you! Please
              fill out the form below, and we'll get back to you as soon as possible.
            </p>
          </ContactUsIntroText>

          {flashMessages.length > 0 &&
            flashMessages.map((msg, idx) => (
              <div key={idx} className={`alert alert-${msg.category}`} role="alert">
                {msg.text}
                <button type="button" className="close" onClick={() => setFlashMessages([])}>
                  &times;
                </button>
              </div>
            ))}

          <ContactUsForm onSubmit={handleFormSubmit} />
        </ContactUsCardBody>
      </ContactUsCard>
    </ContactUsContainer>
  );
};

export default ContactUsTemplate;
