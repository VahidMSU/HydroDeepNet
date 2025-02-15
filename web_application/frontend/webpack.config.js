module.exports = {
  devServer: {
    allowedHosts: ['.localhost', '.ciwre-bae.campusad.msu.edu', '0.0.0.0'],
    host: '0.0.0.0',
    port: 3000, // Default React dev server port
    proxy: {
      '/api': 'http://127.0.0.1:5050', // Proxy API requests to Flask
    },
  },
};
