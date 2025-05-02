// Web Worker for processing station points without blocking the main thread

onmessage = function (e) {
  const { stationPoints, symbolProperties, action } = e.data;

  if (action === 'process') {
    console.log(`[Worker] Processing ${stationPoints.length} station points`);

    try {
      // Process station points in batches
      const BATCH_SIZE = 1000;
      const results = [];

      for (let i = 0; i < stationPoints.length; i += BATCH_SIZE) {
        const batch = stationPoints.slice(i, i + BATCH_SIZE);

        // Process each point in the batch
        const processedBatch = batch.map((point) => {
          // Extract only what we need
          return {
            geometry: point.geometry,
            attributes: {
              SiteNumber: point.properties.SiteNumber,
              SiteName: point.properties.SiteName,
              id: point.properties.id,
            },
          };
        });

        results.push(...processedBatch);

        // Report progress periodically
        if (i % 5000 === 0 && i > 0) {
          postMessage({
            type: 'progress',
            processed: i,
            total: stationPoints.length,
          });
        }
      }

      // Send the processed results back to the main thread
      postMessage({
        type: 'complete',
        results: results,
      });
    } catch (error) {
      postMessage({
        type: 'error',
        message: error.message,
      });
    }
  }
};
