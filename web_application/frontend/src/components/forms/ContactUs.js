import React, { useState } from 'react';
import { Box, Typography, TextField, Checkbox, FormControlLabel, Button, Alert, IconButton, InputLabel } from '@mui/material';
import { styled } from '@mui/system';
import CloseIcon from '@mui/icons-material/Close';

// Styled Components
const ContactContainer = styled(Box)({
  maxWidth: '800px',
  margin: '2rem auto',
  padding: '2rem',
  color: '#ffffff',
  textAlign: 'center',
});

const ContactTitle = styled(Typography)({
  color: 'white',
  fontSize: '4rem',
  marginBottom: '1.5rem',
  paddingBottom: '1.2rem',
  position: 'relative',
  fontWeight: 'bold',
  '&:after': {
    content: '""',
    position: 'absolute',
    bottom: '-3px',
    left: '50%',
    transform: 'translateX(-50%)',
    width: '60px',
    height: '3px',
    backgroundColor: '#ffa533',
  },
});

const ContentWrapper = styled(Box)({
  backgroundColor: '#444e5e',
  borderRadius: '16px',
  padding: '2rem',
  boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)',
  textAlign: 'left',
});

const SubmitButton = styled(Button)({
  backgroundColor: '#e67500',
  color: '#ffffff',
  padding: '0.6rem 1.2rem',
  borderRadius: '8px',
  fontSize: '1.1rem',
  fontWeight: 600,
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  '&:hover': {
    backgroundColor: '#ff8500',
    transform: 'translateY(-2px)',
    boxShadow: '0 5px 15px rgba(255, 133, 0, 0.3)',
  },
});

const CloseButton = styled(IconButton)({
  color: '#fff',
  backgroundColor: '#000',
  padding: '4px',
  borderRadius: '4px',
  '&:hover': {
    backgroundColor: '#333',
  },
});

const ContactUsForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: '',
    newsletter: false,
  });

  const [flashMessage, setFlashMessage] = useState(null); // Store a single flash message

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
    setFlashMessage({ category: 'success', text: 'Your message has been sent successfully!' });

    // Reset the form fields
    setFormData({ name: '', email: '', message: '', newsletter: false });
  };

  return (
    <ContactContainer component="form" onSubmit={handleSubmit}>
      <ContactTitle variant="h2">Contact Us</ContactTitle>

      {/* Display Alert */}
      {flashMessage && (
        <Alert
          severity={flashMessage.category}
          action={
            <CloseButton onClick={() => setFlashMessage(null)}>
              <CloseIcon />
            </CloseButton>
          }
          sx={{ mb: 2 }}
        >
          {flashMessage.text}
        </Alert>
      )}

      <ContentWrapper>
        <TextField
          label="Name"
          id="name"
          name="name"
          variant="outlined"
          fullWidth
          required
          value={formData.name}
          onChange={handleChange}
          slotProps={{
            inputLabel: {
              sx: {
                color: "#2b2b2c",
                bgcolor: 'white',
                px: 1,
                borderRadius: 1,
                '&.MuiInputLabel-shrink': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
              },
            },
          }}
          sx={{ mb: 2, backgroundColor: '#ffffff', borderRadius: '8px', input: { color: '#2b2b2c' } }}
        />

        <TextField
          label="Email"
          id="email"
          name="email"
          type="email"
          variant="outlined"
          fullWidth
          required
          value={formData.email}
          onChange={handleChange}
          slotProps={{
            inputLabel: {
              sx: {
                color: "#2b2b2c",
                bgcolor: 'white',
                px: 1,
                borderRadius: 1,
                '&.MuiInputLabel-shrink': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
              },
            },
          }}
          sx={{ mb: 2, backgroundColor: '#ffffff', borderRadius: '8px', input: { color: '#2b2b2c' } }}
        />

        <TextField
          label="Message"
          id="message"
          name="message"
          variant="outlined"
          multiline
          rows={5}
          fullWidth
          required
          value={formData.message}
          onChange={handleChange}
          slotProps={{
            inputLabel: {
              sx: {
                color: "#2b2b2c",
                bgcolor: 'white',
                px: 1,
                borderRadius: 1,
                '&.MuiInputLabel-shrink': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
              },
            },
          }}
          sx={{ mb: 2, backgroundColor: '#ffffff', borderRadius: '8px', input: { color: '#2b2b2c' } }}
        />

        {/* Subscription Checkbox (Own Line, Left-Aligned) */}
        <Box sx={{ mb: 2 }}>
          <FormControlLabel
            control={<Checkbox id="newsletter" name="newsletter" checked={formData.newsletter} onChange={handleChange} sx={{ color: '#ffa533' }} />}
            label="Subscribe to our newsletter"
            sx={{ color: '#ffffff' }}
          />
        </Box>

        {/* Centered Submit Button */}
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <SubmitButton type="submit" variant="contained">Submit</SubmitButton>
        </Box>
      </ContentWrapper>
    </ContactContainer>
  );
};

export default ContactUsForm;