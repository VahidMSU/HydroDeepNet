import React from 'react';
import '../../styles/TermsConditions.tsx'; // Import the new TSX file

const TermsAndConditionsTemplate = () => {
  const bodyStyle = {
    fontFamily: 'Arial, sans-serif',
    margin: '40px',
    padding: '20px',
    backgroundColor: '#f9f9f9',
    color: '#333',
    lineHeight: '1.6',
  };

  const headerStyle = {
    background: '#004471',
    color: 'white',
    padding: '15px',
    textAlign: 'center',
    borderRadius: '5px',
  };

  const mainStyle = {
    background: 'white',
    padding: '20px',
    borderRadius: '5px',
    boxShadow: '0px 0px 10px rgba(0, 0, 0, 0.1)',
  };

  return (
    <div style={bodyStyle}>
      <header style={headerStyle}>
        <h1>Terms and Conditions</h1>
      </header>
      <main style={mainStyle}>
        <p>
          This web application is provided as a research collaboration tool for hydrological and
          environmental modeling. By accessing and using this platform, you agree to the following
          terms.
        </p>
        <p>
          The application is built on open-source technologies and integrates datasets from public
          sources, including the National Solar Radiation Database (NSRDB), NHDPlus High Resolution,
          LANDFIRE, STATSGO2, and SNODAS. The platform also utilizes software such as QSWAT+,
          SWATPlusEditor, SWAT+, MODFLOW, and FloPy for modeling workflows.
        </p>
        <p>
          Users are responsible for ensuring that their activities comply with Michigan State
          University’s IT policies and relevant research agreements. Unauthorized access, data
          scraping, or any form of misuse, including attempts to bypass authentication or disrupt
          system functionality, is strictly prohibited.
        </p>
        <p>
          Access to specific datasets may be subject to licensing agreements or institutional
          policies. Users contributing data must ensure they have the appropriate permissions to
          share it on this platform.
        </p>
      </main>
    </div>
  );
};

export default TermsAndConditionsTemplate;
