const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  // API requests proxy
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5050',
      changeOrigin: true,
      pathRewrite: {
        '^/api': '/api', // no rewrite needed
      },
      onError: (err, req, res) => {
        console.log('Proxy Error:', err);
        res.status(500).send('Backend API server is not available');
      },
    }),
  );

  // Static images proxy
  app.use(
    '/static/images',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5050',
      changeOrigin: true,
      onError: (err, req, res) => {
        // Fallback to local static files if backend is unavailable
        console.log(`Proxy Error for ${req.path}:`, err);
        const localPath = `/data/SWATGenXApp/GenXAppData${req.path.replace('/static', '')}`;
        console.log('Falling back to local path:', localPath);
      },
    }),
  );
};
