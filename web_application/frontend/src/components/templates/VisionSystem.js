import React, { useState, useRef, useEffect } from 'react';
import {
  ContainerFluid,
  VideoContainer,
  CardBody,
  VideoGrid,
  TextCenter,
  MediaModal,
  MediaWrapper,
  NavigationButton,
  CloseButton,
} from '../../styles/VisionSystem.tsx'; // Import styled components

import {
  Card,
  HeaderTitle,
  Paragraph,
  ImageCard,
  Modal,
  ModalClose,
  Section,
  List,
  SubHeader,
} from '../../styles/Layout.tsx'; // Import styled components

const VisionSystemTemplate = () => {
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState('');
  const videoRefs = useRef([]);
  const mainVideoRef = useRef(null);
  const [selectedMedia, setSelectedMedia] = useState(null);
  const [mediaType, setMediaType] = useState(null); // 'video' or 'image'
  const [currentIndex, setCurrentIndex] = useState(0);

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

  const allMedia = [
    {
      type: 'video',
      src:
        process.env.REACT_APP_PUBLIC_URL +
        '/static/videos/DeepLearningMichiganET_CNNTransformer_reencoded.mp4',
    },
    ...batches.map((batch) => ({
      type: 'video',
      src:
        process.env.REACT_APP_PUBLIC_URL +
        `/static/videos/predictions_vs_ground_truth_batch_${batch}_0_fixed.mp4`,
    })),
    {
      type: 'image',
      src: process.env.REACT_APP_PUBLIC_URL + '/static/images/cell_wise_nse_performance.png',
    },
    { type: 'image', src: process.env.REACT_APP_PUBLIC_URL + '/static/images/CNN-Transformer.png' },
  ];

  const handleMediaClick = (src, type, index) => {
    setSelectedMedia(src);
    setMediaType(type);
    setCurrentIndex(index);
  };

  const handleKeyPress = (e) => {
    if (!selectedMedia) {
      return;
    }

    switch (e.key) {
      case 'Escape':
        setSelectedMedia(null);
        break;
      case 'ArrowLeft':
        setCurrentIndex((prev) => (prev > 0 ? prev - 1 : allMedia.length - 1));
        break;
      case 'ArrowRight':
        setCurrentIndex((prev) => (prev < allMedia.length - 1 ? prev + 1 : 0));
        break;
    }
  };

  const setupVideoSync = () => {
    if (mainVideoRef.current) {
      mainVideoRef.current.addEventListener('play', () => {
        videoRefs.current.forEach((video) => {
          if (video && video !== mainVideoRef.current) {
            video.currentTime = mainVideoRef.current.currentTime;
            video.play();
          }
        });
      });

      mainVideoRef.current.addEventListener('pause', () => {
        videoRefs.current.forEach((video) => {
          if (video && video !== mainVideoRef.current) {
            video.pause();
          }
        });
      });
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectedMedia]);

  useEffect(() => {
    if (selectedMedia) {
      const newMedia = allMedia[currentIndex];
      setSelectedMedia(newMedia.src);
      setMediaType(newMedia.type);
    }
  }, [currentIndex]);

  useEffect(() => {
    setupVideoSync();
    // Start all videos automatically
    videoRefs.current.forEach((video) => {
      if (video) {
        video.play();
      }
    });
  }, []);

  // Update video rendering
  const renderVideo = (src, index) => (
    <VideoContainer onClick={() => handleMediaClick(src, 'video', index)}>
      <video ref={(el) => (videoRefs.current[index] = el)} loop muted playsInline autoPlay>
        <source src={src} type="video/mp4" />
      </video>
    </VideoContainer>
  );

  return (
    <ContainerFluid>
      <HeaderTitle>Vision System Deep Learning</HeaderTitle>

      {/* Introduction & Showcasing Results */}
      <Section>
        <Card>
          <CardBody>
            <Paragraph>
              GeoCNN is a streamlined deep learning framework designed for hydrological modeling. It
              processes large-scale spatiotemporal data—integrating remote sensing, climate, and
              soil/land cover inputs—to predict monthly water balance components (e.g., ET) across
              the Michigan Lower Peninsula.
            </Paragraph>
            <TextCenter>
              <VideoContainer>
                <video ref={mainVideoRef} loop muted playsInline autoPlay>
                  <source
                    src={
                      process.env.REACT_APP_PUBLIC_URL +
                      '/static/videos/DeepLearningMichiganET_CNNTransformer_reencoded.mp4'
                    }
                    type="video/mp4"
                  />
                  Your browser does not support the video tag.
                </video>
              </VideoContainer>
            </TextCenter>
            <TextCenter>
              <SubHeader>Predictions vs. Ground Truth (Single Batches)</SubHeader>
              <VideoGrid>
                {batches.map((batch, index) =>
                  renderVideo(
                    process.env.REACT_APP_PUBLIC_URL +
                      `/static/videos/predictions_vs_ground_truth_batch_${batch}_0_fixed.mp4`,
                    index,
                  ),
                )}
              </VideoGrid>
              <SubHeader className="mt-4">Cell-wise NSE Performance</SubHeader>
              <ImageCard>
                <img
                  src={
                    process.env.REACT_APP_PUBLIC_URL +
                    '/static/images/cell_wise_nse_performance.png'
                  }
                  alt="Cell-wise NSE Performance"
                  className="img-fluid clickable-image"
                  onClick={() =>
                    handleMediaClick(
                      process.env.REACT_APP_PUBLIC_URL +
                        '/static/images/cell_wise_nse_performance.png',
                      'image',
                      batches.length,
                    )
                  }
                />
              </ImageCard>
            </TextCenter>
          </CardBody>
        </Card>
      </Section>

      {/* Key Components & Overview */}
      <Section>
        <SubHeader>Overview of GeoCNN</SubHeader>
        <Card>
          <CardBody>
            <Paragraph>GeoCNN efficiently handles:</Paragraph>
            <List>
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
            </List>
            <Paragraph>
              Target variables (e.g., ET) are observed at 250m resolution across Michigan. Data gaps
              are filled using water masks, mean interpolation, and global scaling from 0 to 1.
              Invalid areas (like Lake Michigan) are marked to help the model distinguish land vs.
              water.
            </Paragraph>
          </CardBody>
        </Card>
      </Section>

      {/* Data Pipeline */}
      <Section>
        <SubHeader>Data Pipeline</SubHeader>
        <Card>
          <CardBody>
            <Paragraph>
              GeoCNN’s data pipeline handles monthly raster stacks spanning 2001–2021. Temporal
              aggregation (summing precipitation, averaging temperature) ensures consistency. A
              70/20/10 split (train/validate/test) provides robust coverage for model development.
            </Paragraph>
            <Paragraph>
              Dynamic variables (e.g., NDVI, EVI, precipitation, temperature) and static features
              (e.g., soil/land cover) are ingested in batches via efficient queue management,
              minimizing latency during training.
            </Paragraph>
          </CardBody>
        </Card>
      </Section>

      {/* Deep Learning Models */}
      <Section>
        <SubHeader>Deep Learning Models</SubHeader>
        <Card>
          <CardBody>
            <List>
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
            </List>
            <Paragraph>
              Multiple iterations (V1–V8) refined hyperparameters (positional encodings, attention
              mechanisms, learning rates, and more). Removing batch normalization from most layers
              helped stabilize training, except in squeeze-and-excitation blocks.
            </Paragraph>
          </CardBody>
        </Card>
      </Section>

      {/* CNN-Transformer Architecture */}
      <Section>
        <SubHeader>CNN-Transformer Architecture</SubHeader>
        <Card>
          <CardBody>
            <Paragraph>
              Our final CNN-Transformer processes inputs of shape [B, T, C, H, W]—batch, time steps,
              channels, height, width. Key elements include:
            </Paragraph>
            <List>
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
            </List>
            <TextCenter>
              <SubHeader>CNN-Transformer Architecture Flow</SubHeader>
              <ImageCard>
                <img
                  src={process.env.REACT_APP_PUBLIC_URL + '/static/images/CNN-Transformer.png'}
                  alt="CNN-Transformer Architecture"
                  className="img-fluid clickable-image"
                  onClick={() =>
                    handleMediaClick(
                      process.env.REACT_APP_PUBLIC_URL + '/static/images/CNN-Transformer.png',
                      'image',
                      batches.length + 1,
                    )
                  }
                />
              </ImageCard>
            </TextCenter>
          </CardBody>
        </Card>
      </Section>

      {/* Specialized Loss Function */}
      <Section>
        <SubHeader>Specialized Loss Function</SubHeader>
        <Card>
          <CardBody>
            <Paragraph>GeoCNN uses a custom loss that evaluates:</Paragraph>
            <List>
              <li>
                <strong>Spatial Dimensions:</strong> Penalizes boundary errors, outliers, and
                no-value regions.
              </li>
              <li>
                <strong>Temporal Dimensions:</strong> Seasonal performance in winter, spring,
                summer, and fall.
              </li>
            </List>
            <Paragraph>
              This ensures that the model captures both local spatial structures and broader
              seasonal trends.
            </Paragraph>
          </CardBody>
        </Card>
      </Section>

      {/* Modal for enlarged images */}
      {modalOpen && (
        <Modal onClick={closeModal}>
          <ModalClose onClick={closeModal}>&times;</ModalClose>
          <img
            id="modalImage"
            src={modalImage}
            alt="Large view"
            style={{ maxWidth: '90%', maxHeight: '90%', borderRadius: '10px' }}
          />
        </Modal>
      )}

      {selectedMedia && (
        <MediaModal>
          <CloseButton onClick={() => setSelectedMedia(null)}>&times;</CloseButton>
          <NavigationButton
            className="prev"
            onClick={() => setCurrentIndex((prev) => (prev > 0 ? prev - 1 : allMedia.length - 1))}
          >
            &#8592;
          </NavigationButton>
          <MediaWrapper>
            {mediaType === 'video' ? (
              <video controls autoPlay loop>
                <source src={selectedMedia} type="video/mp4" />
              </video>
            ) : (
              <img src={selectedMedia} alt="Enlarged view" />
            )}
          </MediaWrapper>
          <NavigationButton
            className="next"
            onClick={() => setCurrentIndex((prev) => (prev < allMedia.length - 1 ? prev + 1 : 0))}
          >
            &#8594;
          </NavigationButton>
        </MediaModal>
      )}
    </ContainerFluid>
  );
};

export default VisionSystemTemplate;
