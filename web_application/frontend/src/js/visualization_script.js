/**
 * Dynamically adds an image to the specified container.
 * @param {string} containerId - The ID of the container.
 * @param {string} imageUrl - The URL of the image.
 * @param {string} altText - Alt text for the image.
 */
function addImage(containerId, imageUrl, altText) {
  const container = document.getElementById(containerId);
  if (!container) {
    return console.error(`Container with ID "${containerId}" not found.`);
  }

  const img = document.createElement('img');
  img.src = imageUrl;
  img.alt = altText;
  img.className = 'visualization-image';
  container.appendChild(img);
}

/**
 * Populates dropdowns with data fetched from the server.
 */
async function populateDropdowns() {
  const controller = new AbortController(); // To handle fetch cancellation
  const { signal } = controller;

  try {
    const response = await fetch('/get_options', { signal, method: 'GET' }); // Ensure this matches the backend method
    if (!response.ok) {
      throw new Error(`Error ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    if (!data.names || !data.variables) {
      throw new Error("Invalid response structure. Expected 'names' and 'variables'.");
    }

    updateDropdown('NAME', data.names, 'Select a Watershed');
    updateDropdown('variable', data.variables, '', true);
  } catch (error) {
    if (error.name !== 'AbortError') {
      console.error('Error fetching dropdown options:', error.message);
      toggleError('Unable to fetch dropdown options. Please try again later.');
    }
  } finally {
    // Cleanup if necessary
  }
}

/**
 * Updates a dropdown with provided options.
 * @param {string} dropdownId - The ID of the dropdown element.
 * @param {string[]} options - The array of options to populate.
 * @param {string} defaultOption - The default option text.
 * @param {boolean} [isMultiple=false] - Whether the dropdown allows multiple selections.
 */
function updateDropdown(dropdownId, options, defaultOption, isMultiple = false) {
  const dropdown = document.getElementById(dropdownId);
  if (!dropdown) {
    return console.error(`Dropdown with ID "${dropdownId}" not found.`);
  }

  dropdown.innerHTML = isMultiple ? '' : `<option value="">${defaultOption}</option>`;

  const fragment = document.createDocumentFragment();
  options.forEach((option) => {
    const opt = document.createElement('option');
    opt.value = option;
    opt.textContent = option;
    fragment.appendChild(opt);
  });

  dropdown.appendChild(fragment);
}

/**
 * Updates visualizations based on user input.
 */
async function updateVisualizations() {
  const name = document.getElementById('NAME').value;
  const version = document.getElementById('ver').value;
  const variables = Array.from(document.getElementById('variable').selectedOptions).map(
    (opt) => opt.value,
  );

  if (!name || !version || variables.length === 0) {
    toggleError('Please complete all required fields.');
    return;
  }

  const params = new URLSearchParams({
    NAME: name,
    ver: version,
    variable: variables.join(','),
  });

  try {
    toggleLoading(true); // Show loading state

    const response = await fetch(`/visualizations?${params.toString()}`, {
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
    });
    if (!response.ok) {
      throw new Error(`Error ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    if (data.error) {
      toggleError(data.error);
      return;
    }

    updateVisualizationResults(data.gif_files || []);
  } catch (error) {
    console.error('Error updating visualizations:', error.message);
    toggleError('An error occurred while fetching visualizations. Please try again.');
  } finally {
    toggleLoading(false); // Hide loading state
  }
}

/**
 * Updates the visualization results section.
 * @param {string[]} gifFiles - Array of GIF file URLs to display.
 */
function updateVisualizationResults(gifFiles) {
  const gifContainer = document.getElementById('gif_container');
  if (!gifContainer) {
    return console.error('GIF container not found.');
  }

  gifContainer.innerHTML = ''; // Clear previous content
  if (gifFiles.length > 0) {
    const fragment = document.createDocumentFragment();
    gifFiles.forEach((gif, index) => {
      const img = document.createElement('img');
      img.src = gif;
      img.alt = `GIF Animation ${index + 1}`;
      img.className = 'visualization-image';
      fragment.appendChild(img);
    });
    gifContainer.appendChild(fragment);
    document.getElementById('visualization_results').classList.remove('d-none');
  } else {
    document.getElementById('visualization_results').classList.add('d-none');
  }

  toggleError(); // Clear any existing errors
}

/**
 * Toggles the error message visibility.
 * @param {string} [message=""] - The error message to display. If empty, hides the error.
 */
function toggleError(message = '') {
  const errorMessage = document.getElementById('error_message');
  if (!errorMessage) {
    return;
  }

  if (message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('d-none');
    errorMessage.focus(); // Accessibility improvement
  } else {
    errorMessage.classList.add('d-none');
    errorMessage.textContent = '';
  }
}

/**
 * Toggles the loading state of the page.
 * @param {boolean} isLoading - Whether the page is in a loading state.
 */
function toggleLoading(isLoading) {
  const submitButton = document.getElementById('show_visualizations');
  if (submitButton) {
    submitButton.disabled = isLoading;
    submitButton.textContent = isLoading ? 'Loading...' : 'Show Visualizations';
  }
}

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
  populateDropdowns();

  const showVisualizationsButton = document.getElementById('show_visualizations');
  if (showVisualizationsButton) {
    showVisualizationsButton.addEventListener('click', updateVisualizations);
  }
});
