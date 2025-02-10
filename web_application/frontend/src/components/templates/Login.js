import React from 'react';
import LoginForm from '../forms/Login';
import { Container, Card, CardBody, Title, Button } from '../../styles/Login.tsx';

const LoginTemplate = ({ formData, handleChange, handleSubmit, errors }) => {
  return (
    <Container>
      <Card>
        <CardBody>
          <Title>Login</Title>

          {/* MSU NetID Login Button */}
          <Button href="/login?msu_oauth=True">Login with MSU NetID</Button>

          {/* Standard Login Form */}
          <LoginForm
            formData={formData}
            handleChange={handleChange}
            handleSubmit={handleSubmit}
            errors={errors}
          />
        </CardBody>
      </Card>
    </Container>
  );
};

export default LoginTemplate;
