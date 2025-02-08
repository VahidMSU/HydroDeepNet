// pages/VisionSystem.js
import React, { useState } from 'react';
import '../styles/VisionSystem.tsx'; // Adjust the path if necessary

const VisionSystem = () => {
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState('');

  // Open the modal with the given image src.
  const openModal = (src) => {
    setModalImage(src);
    setModalOpen(true);
  };

  // Close the modal.
  const closeModal = () => {
    setModalOpen(false);
    setModalImage('');
  };

  // Batch numbers for the prediction videos (1 through 6).
  const batches = [1, 2, 3, 4, 5, 6];

  return (
    <>
      <div className="container mt-4">
        <h1 className="mb-4 text-center">Vision System Deep Learning</h1>

        {/* Introduction & Showcasing Results */}
        <div className="card mb-4">
          <div className="card-body">
            <p className="lead">
              GeoCNN is a streamlined deep learning framework designed for hydrological modeling. It
              processes large-scale spatiotemporal data—integrating remote sensing, climate, and
              soil/land cover inputs—to predict monthly water balance components (e.g., ET) across
              the Michigan Lower Peninsula.
            </p>
            <div className="text-center mt-4">
              <video controls autoPlay className="video-container">
                <source
                  src={
                    process.env.REACT_APP_PUBLIC_URL +
                    '/static/videos/DeepLearningMichiganET_CNNTransformer_reencoded.mp4'
                  }
                  type="video/mp4"
                />
                Your browser does not support the video tag.
              </video>
            </div>
            <div className="text-center mt-4">
              <h5>Predictions vs. Ground Truth (Single Batches)</h5>
              <div className="video-grid">
                {batches.map((batch) => (
                  <video key={batch} controls autoPlay className="video-container small-video">
                    <source
                      src={
                        process.env.REACT_APP_PUBLIC_URL +
                        `/static/videos/predictions_vs_ground_truth_batch_${batch}_0_fixed.mp4`
                      }
                      type="video/mp4"
                    />
                    Your browser does not support the video tag.
                  </video>
                ))}
              </div>
              <h5 className="mt-4">Cell-wise NSE Performance</h5>
              <img
                src={
                  process.env.REACT_APP_PUBLIC_URL + '/static/images/cell_wise_nse_performance.png'
                }
                alt="Cell-wise NSE Performance"
                className="img-fluid clickable-image"
                onClick={() =>
                  openModal(
                    process.env.REACT_APP_PUBLIC_URL +
                      '/static/images/cell_wise_nse_performance.png',
                  )
                }
              />
            </div>
          </div>
        </div>

        {/* Key Components & Overview */}
        <h2 className="mt-4">Overview of GeoCNN</h2>
        <div className="card mb-4">
          <div className="card-body">
            <p>GeoCNN efficiently handles:</p>
            <ul>
              <li>
                <strong>Large Datasets:</strong> Fast loading/reloading of multi-year monthly data.
              </li>
              <li>
                <strong>Multi-Scale Inputs:</strong> Incorporates MODIS, PRISM, soil, and land cover
                info.
              </li>
              <li>
                <strong>Powerful Architectures:</strong> CNN-Transformers and fully
                Transformer-based models.
              </li>
              <li>
                <strong>Spatial & Temporal Flexibility:</strong> Custom skip connections, attention,
                and advanced up/down-sampling.
              </li>
            </ul>
            <p>
              Target variables (e.g., ET) are observed at 250m resolution across Michigan. Data gaps
              are filled using water masks, mean interpolation, and global scaling from 0 to 1.
              Invalid areas (like Lake Michigan) are marked to help the model distinguish land vs.
              water.
            </p>
          </div>
        </div>

        {/* Data Pipeline */}
        <h2 className="mt-4">Data Pipeline</h2>
        <div className="card mb-4">
          <div className="card-body">
            <p>
              GeoCNN’s data pipeline handles monthly raster stacks spanning 2001–2021. Temporal
              aggregation (summing precipitation, averaging temperature) ensures consistency. A
              70/20/10 split (train/validate/test) provides robust coverage for model development.
            </p>
            <p>
              Dynamic variables (e.g., NDVI, EVI, precipitation, temperature) and static features
              (e.g., soil/land cover) are ingested in batches via efficient queue management,
              minimizing latency during training.
            </p>
          </div>
        </div>

        {/* Deep Learning Models */}
        <h2 className="mt-4">Deep Learning Models</h2>
        <div className="card mb-4">
          <div className="card-body">
            <ul>
              <li>
                <strong>Inception-LSTM:</strong> Combines spatial feature extraction with LSTM-based
                temporal modeling.
              </li>
              <li>
                <strong>CNN-Transformers:</strong> Uses a CNN down-sampling path and a Transformer
                encoder for time steps.
              </li>
              <li>
                <strong>Fully Transformer-Based:</strong> Omits convolutional operations entirely,
                focusing on spatiotemporal attention.
              </li>
            </ul>
            <p>
              Multiple iterations (V1–V8) refined hyperparameters (positional encodings, attention
              mechanisms, learning rates, and more). Removing batch normalization from most layers
              helped stabilize training, except in squeeze-and-excitation blocks.
            </p>
          </div>
        </div>

        {/* CNN-Transformer Architecture */}
        <h2 className="mt-4">CNN-Transformer Architecture</h2>
        <div className="card mb-4">
          <div className="card-body">
            <p>
              Our final CNN-Transformer processes inputs of shape [B, T, C, H, W]—batch, time steps,
              channels, height, width. Key elements include:
            </p>
            <ul>
              <li>
                <strong>Down-Sampling Pathway:</strong> Deformable convolutions, SE blocks,
                coordinate attention for hierarchical spatial features.
              </li>
              <li>
                <strong>Transformer Encoder:</strong> Fourier-based positional encoding and
                multi-head attention for temporal modeling.
              </li>
              <li>
                <strong>Up-Sampling Pathway:</strong> Sub-pixel convolution and skip connections to
                restore spatial resolution.
              </li>
            </ul>
            <div className="text-center mt-4">
              <h5>CNN-Transformer Architecture Flow</h5>
              <img
                src={process.env.REACT_APP_PUBLIC_URL + '/static/images/CNN-Transformer.png'}
                alt="CNN-Transformer Architecture"
                className="img-fluid clickable-image"
                onClick={() =>
                  openModal(process.env.REACT_APP_PUBLIC_URL + '/static/images/CNN-Transformer.png')
                }
              />
            </div>
          </div>
        </div>

        {/* Specialized Loss Function */}
        <h2 className="mt-4">Specialized Loss Function</h2>
        <div className="card mb-4">
          <div className="card-body">
            <p>GeoCNN uses a custom loss that evaluates:</p>
            <ul>
              <li>
                <strong>Spatial Dimensions:</strong> Penalizes boundary errors, outliers, and
                no-value regions.
              </li>
              <li>
                <strong>Temporal Dimensions:</strong> Seasonal performance in winter, spring,
                summer, and fall.
              </li>
            </ul>
            <p>
              This ensures that the model captures both local spatial structures and broader
              seasonal trends.
            </p>
          </div>
        </div>
      </div>

      {/* Modal for enlarged images */}
      {modalOpen && (
        <div
          id="imageModal"
          style={{
            display: 'flex',
            position: 'fixed',
            zIndex: 1000,
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            justifyContent: 'center',
            alignItems: 'center',
          }}
          onClick={closeModal}
        >
          <span
            style={{
              position: 'absolute',
              top: '20px',
              right: '30px',
              color: 'white',
              fontSize: '2rem',
              cursor: 'pointer',
            }}
            onClick={closeModal}
          >
            &times;
          </span>
          <img
            id="modalImage"
            src={modalImage}
            alt="Large view"
            style={{ maxWidth: '90%', maxHeight: '90%', borderRadius: '10px' }}
          />
        </div>
      )}
    </>
  );
};

export default VisionSystem;
