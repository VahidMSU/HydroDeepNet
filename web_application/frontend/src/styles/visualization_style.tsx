/** @jsxImportSource @emotion/react */
import { css } from '@emotion/react';
import styled from '@emotion/styled';
import { Button, TextField, Select, MenuItem, Container, Box, Typography } from '@mui/material';

const bodyStyle = css`
  background-color: #f0f2f5;
  color: #1a1a1a;
  line-height: 1.6;
  text-size-adjust: 100%;
`;

const headingStyle = css`
  font-weight: 700;
  margin-bottom: 1.5rem;
  color: #1a1a1a;
  letter-spacing: -0.5px;
  text-align: match-parent;
`;

const FormContainer = styled(Box)`
  background-color: #ffffff;
  border-radius: 16px;
  padding: 2.5rem;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
  margin: 2rem auto;
  max-width: 1200px;
`;

const FormLabel = styled(Typography)`
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  font-size: 0.9rem;
  color: #4a5568;
`;

const FormControl = styled(TextField)`
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.2s ease;
  background-color: #f8fafc;
  margin-bottom: 1rem;

  &:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    outline: none;
  }
`;

const FormSelect = styled(Select)`
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.2s ease;
  background-color: #f8fafc;
  margin-bottom: 1rem;

  &:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    outline: none;
  }
`;

const FormGrid = styled(Box)`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
  backdrop-filter: blur(10px);
`;

const FormGroup = styled(Box)`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const ButtonPrimary = styled(Button)`
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.2s ease;
  border: none;
  cursor: pointer;
  background-color: #3b82f6;
  color: white;

  &:hover {
    background-color: #2563eb;
    transform: translateY(-1px);
  }
`;

const ButtonSecondary = styled(Button)`
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.2s ease;
  border: none;
  cursor: pointer;
  background-color: #64748b;
  color: white;

  &:hover {
    background-color: #475569;
    transform: translateY(-1px);
  }
`;

const FormError = styled(Typography)`
  color: #ef4444;
  font-size: 0.875rem;
  margin-top: 0.25rem;
`;

const FormActions = styled(Box)`
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
`;

const VisualizationResults = styled(Box)`
  margin-top: 2rem;
`;

const GifContainer = styled(Box)`
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 1rem;
`;

const GifWrapper = styled(Box)`
  max-width: 300px;
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);

  img {
    width: 100%;
    height: auto;
  }
`;

const ResponsiveDesign = css`
  @media (max-width: 768px) {
    .form-container {
      padding: 1.5rem;
      margin: 1rem;
    }

    .form-actions {
      flex-direction: column;
    }

    .btn {
      width: 100%;
      margin-bottom: 0.5rem;
    }
  }
`;

export {
  bodyStyle,
  headingStyle,
  FormContainer,
  FormLabel,
  FormControl,
  FormSelect,
  FormGrid,
  FormGroup,
  ButtonPrimary,
  ButtonSecondary,
  FormError,
  FormActions,
  VisualizationResults,
  GifContainer,
  GifWrapper,
  ResponsiveDesign,
};
