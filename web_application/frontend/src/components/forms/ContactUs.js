///data/SWATGenXApp/codes/web_application/frontend/src/components/forms/ContactUs.js
import React, { useState } from 'react';
import { FormGroup, Label, Input, TextArea, SubmitButton } from '../../styles/ContactUs.tsx';

const ContactUsForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: '',
    newsletter: false,
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
    setFormData({ name: '', email: '', message: '', newsletter: false });
  };

  return (
    <form onSubmit={handleSubmit}>
      <FormGroup>
        <Label htmlFor="name">Name:</Label>
        <Input
          type="text"
          id="name"
          name="name"
          placeholder="Enter your full name"
          required
          value={formData.name}
          onChange={handleChange}
        />
      </FormGroup>
      <FormGroup>
        <Label htmlFor="email">Email:</Label>
        <Input
          type="email"
          id="email"
          name="email"
          placeholder="Enter your email address"
          required
          value={formData.email}
          onChange={handleChange}
        />
      </FormGroup>
      <FormGroup>
        <Label htmlFor="message">Message:</Label>
        <TextArea
          id="message"
          name="message"
          rows="5"
          placeholder="Write your message here"
          required
          value={formData.message}
          onChange={handleChange}
        />
      </FormGroup>
      <FormGroup>
        <Input
          type="checkbox"
          id="newsletter"
          name="newsletter"
          checked={formData.newsletter}
          onChange={handleChange}
        />
        <Label htmlFor="newsletter" style={{ display: 'inline', marginLeft: '10px' }}>
          Subscribe to our newsletter
        </Label>
      </FormGroup>
      <SubmitButton type="submit">Submit</SubmitButton>
    </form>
  );
};

export default ContactUsForm;
