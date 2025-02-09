import React, { useState } from 'react';
import ContactUsForm from '../forms/ContactUs.js';
import '../../styles/ContactUs.tsx'; // Assuming you have a CSS file for custom styles

const ContactUsTemplate = ({ handleFormSubmit, flashMessages, setFlashMessages }) => {
  return (
    <div className="contact-us-container">
      <h2 className="text-center">Contact Us</h2>
      <div className="card">
        <div className="card-body">
          <p className="intro-text">
            Have questions, feedback, or collaboration ideas? We'd love to hear from you! Please
            fill out the form below, and we'll get back to you as soon as possible.
          </p>

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
        </div>
      </div>
    </div>
  );
};

export default ContactUsTemplate;
