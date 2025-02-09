/** @jsxImportSource @emotion/react */
import styled from '@emotion/styled';

const ContainerFluid = styled.div`
  display: flex;
  min-height: 100vh;
  flex-direction: column;
`;

const Sidebar = styled.div`
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

const Nav = styled.nav`
  display: flex;
  flex-direction: column;
`;

const NavLink = styled(Link)`
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

const ContentWrapper = styled.div`
  flex-grow: 1;
  margin-left: 270px;
  padding: 30px;
  background-color: #ffffff;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  overflow-y: auto;
`;

const Footer = styled.footer`
  width: 100%;
  background-color: #343a40;
  color: #d8dbde;
  text-align: center;
  padding: 15px;
  font-size: 10px;
  border-top: 1px solid #343a40;
  position: fixed;
  bottom: 0;

  a {
    color: #d8dbde;
    text-decoration: none;
    margin: 0 10px;
    font-weight: 500;

    &:hover {
      text-decoration: underline;
    }
  }
`;

const LogoutButton = styled.button`
  color: #adb5bd;
  background: none;
  border: none;
  text-align: left;
  padding: 0;
  margin: 15px 0;
  display: flex;
  align-items: center;
  cursor: pointer;

  &:hover {
    color: #ffffff;
  }

  i {
    margin-right: 10px;
  }
`;

export { ContainerFluid, Sidebar, Nav, NavLink, ContentWrapper, Footer, LogoutButton };
