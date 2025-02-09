import styled, { createGlobalStyle, keyframes } from 'styled-components';
import '../css/Layout.css'; // Ensure the path is correct

export const GlobalStyle = createGlobalStyle`
  body {
    background-color: #f8f9fa;
    text-size-adjust: 100%;
  }
`;

export const HeaderTitle = styled.h1`
  text-align: center;
  margin-bottom: 30px;
  color: #333;
  font-size: 2.5rem;
  font-weight: bold;
  text-align: -webkit-match-parent;
  text-align: match-parent;
`;




export const Sidebar = styled.nav`
  background-color: #343a40;
  color: white;
  width: 250px;
  padding: 20px;
  position: fixed;
  height: 100%;
  overflow-y: auto;
  transition: all 0.3s;
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);

  h2 {
    color: #ffc107;
    font-weight: bold;
    margin-bottom: 30px;
  }
`;

export const NavLink = styled.a`
  color: #adb5bd;
  margin: 15px 0;
  display: flex;
  align-items: center;
  text-decoration: none;

  &:hover {
    color: #ffffff;
  }

  i {
    margin-right: 10px;
  }
`;



export const ViewDiv = styled.div`
  flex: 2;
  min-width: 200px;
  height: 800px;
  border: 1px solid #ddd;
  border-radius: 0.5rem;
  box-shadow: 0 0.25rem 0.625rem rgba(0, 0, 0, 0.1);
  background-color: #fff;
`;

export const Card = styled.div`
  margin-bottom: 20px;
`;

export const CheckboxContainer = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 10px;

  .form-check-label {
    margin-right: 15px;
  }
`;

export const SettingsContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 15px;

  .form-group {
    margin-bottom: 0;
    flex: 1;
    min-width: 100px;
  }
`;

export const FormGroup = styled.div`
  label {
    font-weight: bold;
    margin-bottom: 5px;
    display: block;
  }
`;

export const StyledImage = styled.img`
  max-width: 100%;
  height: auto;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  margin-top: 20px;
`;

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

export const LoadingIndicator = styled.div`
  img {
    animation: ${spin} 1s linear infinite;
  }
`;

export const Button = styled.button`
  background-color: #007bff;
  color: #fff;
  padding: 10px 20px;
  border-radius: 5px;
  text-decoration: none;
  font-size: 1rem;
  transition: background-color 0.3s;
  border: none;
  cursor: pointer;

  &:hover {
    background-color: #0056b3;
  }
`;

export const ButtonPrimary = styled(Button)`
  background-color: #007bff;
  border-color: #007bff;
  padding: 0.625rem 1.5rem;
  font-weight: 500;
  text-transform: capitalize;
  border-radius: 0.3125rem;
  transition: background-color 0.3s ease, transform 0.3s ease;

  &:hover {
    background-color: #0056b3;
    transform: scale(1.05);
  }

  &:active {
    background-color: #004085;
    border-color: #003768;
  }
`;

export const FormControl = styled.input`
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid #ced4da;
  border-radius: 0.25rem;
  padding: 0.625rem;
  font-size: 0.95rem;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
  margin-bottom: 10px;

  &:focus {
    border-color: #80bdff;
    box-shadow: 0 0 0 0.125rem rgba(0, 123, 255, 0.25);
  }
`;


export const Modal = styled.div`
  display: none;
  position: fixed;
  z-index: 1000;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.8);
  align-items: center;
  justify-content: center;

  img {
    max-width: 90%;
    max-height: 90%;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
  }
`;

export const Section = styled.div`
  margin-bottom: 30px;
`;

export const Header = styled.h3`
  font-size: 1.75rem;
  color: #333;
  margin-bottom: 15px;
`;

export const SubHeader = styled.h4`
  font-size: 1.5rem;
  color: #555;
  margin-bottom: 10px;
`;

export const Paragraph = styled.p`
  font-size: 1rem;
  color: #666;
  line-height: 1.5;
  margin-bottom: 15px;
`;

export const List = styled.ul`
  list-style-type: disc;
  padding-left: 20px;
  margin-bottom: 15px;

  li {
    margin-bottom: 10px;
  }
`;

export const ImageGrid = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
`;

export const ImageCard = styled.div`
  flex: 1;
  min-width: 200px;
  border: 1px solid #ddd;
  border-radius: 10px;
  box-shadow: 0 0.25rem 0.625rem rgba(0, 0, 0, 0.1);
  overflow: hidden;
  text-align: center;

  img {
    width: 100%;
    height: auto;
    cursor: pointer;
  }

  h4 {
    margin: 10px 0;
    font-size: 1.25rem;
    color: #333;
  }
`;

export const InteractiveButtons = styled.div`
  display: flex;
  gap: 10px;
  margin-top: 20px;

  a {
    background-color: #007bff;
    color: #fff;
    padding: 10px 20px;
    border-radius: 5px;
    text-decoration: none;
    font-size: 1rem;
    transition: background-color 0.3s;
    border: none;
    cursor: pointer;

    &:hover {
      background-color: #0056b3;
    }
  }
`;

export const ModalClose = styled.span`
  position: absolute;
  top: 20px;
  right: 30px;
  color: #fff;
  font-size: 40px;
  font-weight: bold;
  cursor: pointer;
`;
