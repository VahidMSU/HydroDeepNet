import React from 'react';
import { bodyStyle, headerStyle, mainStyle, footerStyle } from '../../styles/Privacy.tsx';

const PrivacyTemplate = () => {
 
  return (
    <div style={bodyStyle}>
      <header style={headerStyle}>
        <h1>Privacy Policy</h1>
      </header>

      <main style={mainStyle}>
        <p>
          This web application is designed to support research and collaboration by providing access
          to hydrological modeling tools and datasets. We respect user privacy and are committed to
          protecting any information collected during platform use.
        </p>
        <p>
          We collect user interactions with the application solely for improving system
          functionality, security, and performance. This includes authentication logs and feature
          usage analytics but excludes personally identifiable data beyond what is necessary for
          secure access. All passwords are stored securely using encryption standards.
        </p>
        <p>
          User data will not be shared with third parties unless mandated by Michigan State
          University’s IT security policies or legal requirements. Any data provided for research
          purposes remains under the ownership of the contributing institution or researcher, and
          its use follows applicable agreements.
        </p>
        <p>
          The platform operates entirely on open-source software and integrates national datasets,
          including the National Solar Radiation Database (NSRDB), NHDPlus, LANDFIRE, 3D Elevation
          Program, STATSGO2, and SNODAS. We do not collect or process sensitive personal
          information.
        </p>
        <p>
          By using this application, you acknowledge that minimal system interaction data is
          collected to improve platform reliability. You may contact the administrators for
          questions regarding data policies or security measures.
        </p>
      </main>

      <footer style={footerStyle}>
        <p>&copy; 2025 Michigan State University. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default PrivacyTemplate;
