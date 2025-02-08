// pages/Home.js
import React from 'react';
import '../css/Home.css'; // Corrected CSS file path
import '../css/Layout.css'; // Corrected CSS file path

const Home = () => {
  return (
    <div className="container">
      <h1 className="header-title">Hydrological Modeling and Deep Learning Framework</h1>

      {/* Overview Section */}
      <div className="section">
        <h3>Overview</h3>
        <p>
          This platform integrates advanced hydrological modeling, hierarchical data management, and
          deep learning techniques. By leveraging models such as SWAT+ and MODFLOW, it predicts
          hydrological variables at high spatial and temporal resolutions, enabling improved
          understanding of surface and groundwater processes.
        </p>
      </div>

      {/* Key Components Section */}
      <div className="section">
        <h3>Key Components</h3>

        {/* Hydrological Modeling with SWAT+ */}
        <h4>1. Hydrological Modeling with SWAT+</h4>
        <p>
          SWAT+ serves as the core model for simulating surface and subsurface hydrological cycles.
          It integrates datasets such as NHDPlus HR (1:24k resolution) and DEM (30m resolution) to
          accurately model watershed processes. Key highlights:
        </p>
        <ul>
          <li>Simulates evapotranspiration, runoff, and groundwater recharge.</li>
          <li>Uses hierarchical land classification for HRU-based analysis.</li>
          <li>
            Employs Particle Swarm Optimization (PSO) for calibrating hydrological parameters.
          </li>
        </ul>

        {/* Hierarchical Data Management */}
        <h4>2. Hierarchical Data Management</h4>
        <p>
          The platform uses a robust HDF5 database to manage multi-resolution data, integrating
          datasets like:
        </p>
        <ul>
          <li>Land use and soil data (250m resolution).</li>
          <li>
            Groundwater hydraulic properties derived by Empirical Bayesian Kriging (EBK) from 650k
            water wells observation.
          </li>
          <li>Meteorological inputs from PRISM (4km) and NSRDB (2km, upsampled to 4km).</li>
        </ul>
        <p>
          Temporal and spatial preprocessing ensures consistent resolution and gap-filling for
          missing data.
        </p>

        {/* GeoNet Vision System */}
        <h4>3. GeoNet Vision System</h4>
        <p>
          GeoNet leverages hydrological data to perform spatiotemporal regression tasks. Using
          CNN-Transformers and other deep learning architectures, GeoNet predicts groundwater
          recharge, climate impacts, and more. Features include:
        </p>
        <ul>
          <li>Support for 4D spatiotemporal analysis at 250m resolution.</li>
          <li>
            Efficient processing of hydrological data with specialized loss functions for spatial
            and temporal evaluation.
          </li>
          <li>Modular design for hyperparameter tuning and model customization.</li>
        </ul>
      </div>

      {/* Visual Interactive Section */}
      <div className="section">
        <h3>Hydrological Model Creation Framework</h3>
        <div className="image-grid">
          <div className="image-card">
            <img
              src={`/static/images/SWATGenX_flowchart.jpg`}
              alt="SWATGenX Workflow"
              onClick={() => openModal(`/static/images/SWATGenX_flowchart.jpg`)}
            />
            <h4>SWATGenX Workflow</h4>
          </div>
        </div>
      </div>

      {/* Applications Section */}
      <div className="section">
        <h3>Applications</h3>
        <ul>
          <li>Predicting groundwater recharge in data-scarce regions.</li>
          <li>Assessing the impacts of climate change on hydrological processes.</li>
          <li>Supporting scalable watershed-level hydrological modeling and decision-making.</li>
        </ul>
      </div>

      {/* Interactive Buttons */}
      <div className="interactive-btns">
        <a href="#" className="btn">
          Learn More
        </a>
        <a href="#" className="btn">
          Download Data
        </a>
      </div>

      {/* Modal */}
      <div className="modal" id="imageModal" style={{ display: 'none' }}>
        <span className="modal-close" onClick={closeModal}>
          &times;
        </span>
        <img id="modalImage" src="" alt="Large view" />
      </div>
    </div>
  );
};

// Open the modal with the clicked image
const openModal = (imageSrc) => {
  const modal = document.getElementById('imageModal');
  const modalImage = document.getElementById('modalImage');

  modal.style.display = 'block';
  modalImage.src = imageSrc;
};

// Close the modal
const closeModal = () => {
  const modal = document.getElementById('imageModal');
  modal.style.display = 'none';
};

export default Home;
