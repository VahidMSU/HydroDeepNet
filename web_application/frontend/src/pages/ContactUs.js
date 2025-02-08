import React, { useState } from 'react';
import '../styles/ContactUs.tsx'; // Assuming you have a CSS file for custom styles

const ContactUs = () => {
  const [flashMessages, setFlashMessages] = useState([]);
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
    console.log('Submitted:', formData);
    setFlashMessages([{ category: 'success', text: 'Your message has been sent!' }]);
    setFormData({ name: '', email: '', message: '', newsletter: false });
  };

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

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="name">Name:</label>
              <input
                type="text"
                id="name"
                name="name"
                placeholder="Enter your full name"
                required
                value={formData.name}
                onChange={handleChange}
              />
            </div>

            <div className="form-group">
              <label htmlFor="email">Email:</label>
              <input
                type="email"
                id="email"
                name="email"
                placeholder="Enter your email address"
                required
                value={formData.email}
                onChange={handleChange}
              />
            </div>

            <div className="form-group">
              <label htmlFor="message">Message:</label>
              <textarea
                id="message"
                name="message"
                rows="5"
                placeholder="Write your message here"
                required
                value={formData.message}
                onChange={handleChange}
              ></textarea>
            </div>

            <div className="form-group">
              <input
                type="checkbox"
                id="newsletter"
                name="newsletter"
                checked={formData.newsletter}
                onChange={handleChange}
              />
              <label htmlFor="newsletter">Subscribe to our newsletter</label>
            </div>

            <button type="submit" className="btn-submit">
              Submit
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ContactUs;
