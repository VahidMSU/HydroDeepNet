module.exports = {
  devServer: {
    allowedHosts: ['.localhost', '.swatgenx.com', '0.0.0.0'],
    host: '0.0.0.0',
    port: 3000, // Default React dev server port
    proxy: {
      '/api': 'http://127.0.0.1:5050', // Proxy API requests to Flask
    },
  },
};
