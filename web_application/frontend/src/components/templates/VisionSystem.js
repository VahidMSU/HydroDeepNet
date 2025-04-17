import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import {
  VisionContainer,
  VisionTitle,
  ContentWrapper,
  SectionHeader,
  SectionText,
  VideoGrid,
  VideoContainer,
  MediaModal,
  MediaWrapper,
  NavigationButton,
  CloseButton,
} from '../../styles/VisionSystem.tsx';

const VisionSystemTemplate = () => {
  const videoRefs = useRef([]);
  const mainVideoRef = useRef(null);
  const [selectedMedia, setSelectedMedia] = useState(null);
  const [mediaType, setMediaType] = useState(null); // 'video' or 'image'
  const [currentIndex, setCurrentIndex] = useState(0);

  // Batch numbers for the prediction videos (1 through 6).
  const batches = useMemo(() => [1, 2, 3, 4, 5, 6], []);

  const allMedia = useMemo(
    () => [
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
      {
        type: 'image',
        src: process.env.REACT_APP_PUBLIC_URL + '/static/images/CNN-Transformer.png',
      },
    ],
    [batches],
  );

  const handleMediaClick = (src, type, index) => {
    setSelectedMedia(src);
    setMediaType(type);
    setCurrentIndex(index);
    // Force video to be treated as new source in modal
    if (type === 'video') {
      const videoEl = document.querySelector('.modal-video');
      if (videoEl) {
        videoEl.load();
      }
    }
  };

  useEffect(() => {
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
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectedMedia, allMedia]);

  useEffect(() => {
    if (selectedMedia) {
      const newMedia = allMedia[currentIndex];
      setSelectedMedia(newMedia.src);
      setMediaType(newMedia.type);
    }
  }, [currentIndex, allMedia, selectedMedia]);

  const setupVideoSync = useCallback(() => {
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
  }, []);

  useEffect(() => {
    setupVideoSync();
    // Start all videos automatically
    videoRefs.current.forEach((video) => {
      if (video) {
        video.play();
      }
    });
  }, [setupVideoSync]);

  // Update video rendering with a key prop
  const renderVideo = (src, index) => (
    <VideoContainer key={`video-${index}`} onClick={() => handleMediaClick(src, 'video', index)}>
      <video ref={(el) => (videoRefs.current[index] = el)} loop muted playsInline autoPlay>
        <source src={src} type="video/mp4" />
      </video>
    </VideoContainer>
  );

  return (
    <VisionContainer>
      <VisionTitle>Vision System Deep Learning</VisionTitle>

      <ContentWrapper>
        <section>
          <SectionHeader>Overview</SectionHeader>
          <SectionText>
            GeoCNN is a streamlined deep learning framework designed for hydrological modeling. It
            processes large-scale spatiotemporal data—integrating remote sensing, climate, and
            soil/land cover inputs—to predict monthly water balance components across the Michigan
            Lower Peninsula.
          </SectionText>

          <VideoContainer className="main-video">
            <video
              ref={mainVideoRef}
              loop
              muted
              playsInline
              autoPlay
              onClick={() =>
                handleMediaClick(
                  `${process.env.REACT_APP_PUBLIC_URL}/static/videos/DeepLearningMichiganET_CNNTransformer_reencoded.mp4`,
                  'video',
                  0,
                )
              }
            >
              <source
                src={`${process.env.REACT_APP_PUBLIC_URL}/static/videos/DeepLearningMichiganET_CNNTransformer_reencoded.mp4`}
                type="video/mp4"
              />
            </video>
          </VideoContainer>

          <SectionHeader>Predictions vs. Ground Truth</SectionHeader>
          <VideoGrid>
            {batches.map((batch, index) =>
              renderVideo(
                process.env.REACT_APP_PUBLIC_URL +
                  `/static/videos/predictions_vs_ground_truth_batch_${batch}_0_fixed.mp4`,
                index + 1, // Added +1 to avoid conflict with main video index
              ),
            )}
          </VideoGrid>
        </section>

        <section>
          <SectionHeader>Overview of GeoCNN</SectionHeader>
          <SectionText>GeoCNN efficiently handles:</SectionText>
          <ul>
            <li>
              <strong>Large Datasets:</strong> Fast loading/reloading of multi-year monthly data.
            </li>
            <li>
              <strong>Multi-Scale Inputs:</strong> Incorporates MODIS, PRISM, soil, and land cover
              info.
            </li>
            <li>
              <strong>Powerful Architectures:</strong> CNN-Transformers and fully Transformer-based
              models.
            </li>
            <li>
              <strong>Spatial & Temporal Flexibility:</strong> Custom skip connections, attention,
              and advanced up/down-sampling.
            </li>
          </ul>
        </section>

        <section>
          <SectionHeader>Data Pipeline</SectionHeader>
          <SectionText>
            GeoCNN's data pipeline handles monthly raster stacks spanning 2001–2021. Temporal
            aggregation (summing precipitation, averaging temperature) ensures consistency. A
            70/20/10 split (train/validate/test) provides robust coverage for model development.
            Dynamic variables (e.g., NDVI, EVI, precipitation, temperature) and static features
            (e.g., soil/land cover) are ingested in batches via efficient queue management,
            minimizing latency during training.
          </SectionText>
        </section>

        <section>
          <SectionHeader>Deep Learning Models</SectionHeader>
          <SectionText>
            Our models include:
          </SectionText>
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
        </section>

        <section>
          <SectionHeader>Model Performance</SectionHeader>
          <VideoContainer>
            <img
              src={`${process.env.REACT_APP_PUBLIC_URL}/static/images/cell_wise_nse_performance.png`}
              alt="Cell-wise NSE Performance"
              onClick={() =>
                handleMediaClick(
                  `${process.env.REACT_APP_PUBLIC_URL}/static/images/cell_wise_nse_performance.png`,
                  'image',
                  batches.length,
                )
              }
            />
          </VideoContainer>
        </section>

        <section>
          <SectionHeader>CNN-Transformer Architecture</SectionHeader>
          <SectionText>
            Our final CNN-Transformer processes inputs of shape [B, T, C, H, W]—batch, time steps,
            channels, height, width. Key elements include:
          </SectionText>
          <ul>
            <li>
              <strong>Down-Sampling Pathway:</strong> Deformable convolutions, SE blocks, coordinate
              attention for hierarchical spatial features.
            </li>
            <li>
              <strong>Transformer Encoder:</strong> Fourier-based positional encoding and multi-head
              attention for temporal modeling.
            </li>
            <li>
              <strong>Up-Sampling Pathway:</strong> Sub-pixel convolution and skip connections to
              restore spatial resolution.
            </li>
          </ul>
        </section>

        <section>
          <SectionHeader>Specialized Loss Function</SectionHeader>
          <SectionText>GeoCNN uses a custom loss that evaluates:</SectionText>
          <ul>
            <li>
              <strong>Spatial Dimensions:</strong> Penalizes boundary errors, outliers, and no-value
              regions.
            </li>
            <li>
              <strong>Temporal Dimensions:</strong> Seasonal performance in winter, spring, summer,
              and fall.
            </li>
          </ul>
        </section>

        {selectedMedia && (
          <MediaModal
            onClick={(e) => {
              if (e.target === e.currentTarget) {
                setSelectedMedia(null);
              }
            }}
          >
            <CloseButton onClick={() => setSelectedMedia(null)}>&times;</CloseButton>
            <NavigationButton
              className="prev"
              onClick={() => setCurrentIndex((prev) => (prev > 0 ? prev - 1 : allMedia.length - 1))}
            >
              &#8592;
            </NavigationButton>
            <MediaWrapper>
              {mediaType === 'video' ? (
                <video
                  controls
                  autoPlay
                  loop
                  className="modal-video"
                  key={selectedMedia} // Force remount when source changes
                >
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
      </ContentWrapper>
    </VisionContainer>
  );
};

export default VisionSystemTemplate;
