import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import LoginForm from '../forms/Login'; // Import the new LoginForm component
import 'bootstrap/dist/css/bootstrap.min.css';

const LoginTemplate = ({ formData, handleChange, handleSubmit, errors }) => {
  return (
    <div className="container my-5">
      <div className="row justify-content-center">
        <div className="col-md-6 col-lg-4">
          <div className="card shadow">
            <div className="card-body p-4">
              <h2 className="text-center mb-4">Login</h2>

              {/* MSU NetID Login Button */}
              <div className="d-grid mb-3">
                <a href="/login?msu_oauth=True" className="btn btn-success">
                  Login with MSU NetID
                </a>
              </div>

              {/* Standard Login Form */}
              <LoginForm
                formData={formData}
                handleChange={handleChange}
                handleSubmit={handleSubmit}
                errors={errors}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginTemplate;
