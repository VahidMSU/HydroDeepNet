require([
  "esri/Map",
  "esri/views/MapView",
  "esri/layers/GraphicsLayer",
  "esri/widgets/Sketch",
  "esri/geometry/SpatialReference",
  "esri/geometry/support/webMercatorUtils",
], function (
  Map,
  MapView,
  GraphicsLayer,
  Sketch,
  SpatialReference,
  webMercatorUtils
) {
  // Initialize the GraphicsLayer for interactive drawings
  const graphicsLayer = new GraphicsLayer();

  // Create the Map instance
  const map = new Map({
    basemap: "topo-vector",
    layers: [graphicsLayer],
  });

  // Configure the MapView
  const view = new MapView({
    container: "viewDiv",
    map: map,
    center: [-84.5, 43.4], // Initial center coordinates
    zoom: 7,
    ui: { components: ["attribution"] },
  });

  // Remove the attribution UI component
  view.ui.remove("attribution");

  // Initialize the Sketch widget for drawing
  const sketch = new Sketch({
    layer: graphicsLayer,
    view: view,
    creationMode: "update",
    visibleElements: {
      createTools: {
        point: true,
        polyline: false,
        polygon: true,
        rectangle: true,
        circle: false,
      },
      selectionTools: {
        "rectangle-selection": true,
      },
    },
  });

  // Add the Sketch widget to the top-right corner
  view.ui.add(sketch, "bottom-left");

  // Update fields dynamically when Sketch events occur
  sketch.on("create", (event) => updateFields(event));
  sketch.on("update", (event) => updateFields(event));

  // Update fields dynamically when the user clicks on the map
  view.on("click", function (event) {
    const geographicPoint = webMercatorUtils.webMercatorToGeographic(
      event.mapPoint
    );
    const lat = geographicPoint.latitude.toFixed(6);
    const lon = geographicPoint.longitude.toFixed(6);
    updatePointFields(lat, lon);
  });

  /**
   * Handle geometry updates and populate form fields based on geometry type.
   * @param {object} event - The event object from the Sketch widget.
   */
  function updateFields(event) {
    const geometry = event.graphic?.geometry;
    if (!geometry) return;

    const geographicGeometry =
      webMercatorUtils.webMercatorToGeographic(geometry);

    switch (geometry.type) {
      case "extent":
        updateExtentFields(geographicGeometry);
        break;
      case "polygon":
        updatePolygonFields(geographicGeometry);
        break;
      default:
        console.warn("Unsupported geometry type:", geometry.type);
    }
  }

  /**
   * Update form fields for a point geometry.
   * @param {string} lat - Latitude.
   * @param {string} lon - Longitude.
   */
  function updatePointFields(lat, lon) {
    document.querySelector("#latitude").value = lat;
    document.querySelector("#longitude").value = lon;
  }

  /**
   * Update form fields for an extent geometry.
   * @param {object} extent - Extent geometry in WGS84.
   */
  function updateExtentFields(extent) {
    document.querySelector('[name="min_latitude"]').value =
      extent.ymin.toFixed(6);
    document.querySelector('[name="max_latitude"]').value =
      extent.ymax.toFixed(6);
    document.querySelector('[name="min_longitude"]').value =
      extent.xmin.toFixed(6);
    document.querySelector('[name="max_longitude"]').value =
      extent.xmax.toFixed(6);
  }

  /**
   * Update form fields for a polygon geometry.
   * @param {object} polygon - Polygon geometry in WGS84.
   */
  function updatePolygonFields(polygon) {
    const coordinates = polygon.rings[0];
    const latitudes = coordinates.map(([lon, lat]) => lat);
    const longitudes = coordinates.map(([lon, lat]) => lon);

    document.querySelector('[name="min_latitude"]').value = Math.min(
      ...latitudes
    ).toFixed(6);
    document.querySelector('[name="max_latitude"]').value = Math.max(
      ...latitudes
    ).toFixed(6);
    document.querySelector('[name="min_longitude"]').value = Math.min(
      ...longitudes
    ).toFixed(6);
    document.querySelector('[name="max_longitude"]').value = Math.max(
      ...longitudes
    ).toFixed(6);

    document.querySelector("#polygon_coordinates").value =
      JSON.stringify(coordinates);
  }

  /**
   * Reset all AOI-related form fields.
   */
  function resetAoiFields() {
    document.querySelector("#polygon_coordinates").value = "";
    document.querySelector('[name="min_latitude"]').value = "";
    document.querySelector('[name="max_latitude"]').value = "";
    document.querySelector('[name="min_longitude"]').value = "";
    document.querySelector('[name="max_longitude"]').value = "";
    document.querySelector("#latitude").value = "";
    document.querySelector("#longitude").value = "";
  }

  // Expose reset functionality for external use
  window.resetAoiFields = resetAoiFields;

  // Initialize the variable dropdown
  $(document).ready(function () {
    // Attach change event handler to the variable dropdown
    $("#variable").on("change", function () {
      const selectedVariable = $(this).val();
      const subvariableDropdown = $("#subvariable");
      subvariableDropdown.empty(); // Clear existing options

      if (selectedVariable) {
        // Get the CSRF token from the hidden field
        const csrfToken = $('input[name="csrf_token"]').val();

        // AJAX call to fetch subvariables
        $.ajax({
          url: "/get_subvariables",
          method: "POST",
          headers: { "X-CSRFToken": csrfToken }, // Add CSRF token to headers
          data: { variable: selectedVariable },
          success: function (response) {
            const subvariables = response.subvariables || [];
            if (subvariables.length === 0) {
              subvariableDropdown.append(
                '<option value="">No subvariables available</option>'
              );
            } else {
              // Populate the subvariable dropdown
              subvariables.forEach((subvariable) => {
                subvariableDropdown.append(
                  `<option value="${subvariable}">${subvariable}</option>`
                );
              });
            }
          },
          error: function (xhr, status, error) {
            console.error("Error:", error);
            alert("Failed to load subvariables. Please try again.");
          },
        });
      } else {
        // Add a default option when no variable is selected
        subvariableDropdown.append(
          '<option value="">Select a variable first</option>'
        );
      }
    });

    // Ensure form submission includes correct values
    $("form").on("submit", function () {
      const variable = $("#variable").val();
      const subvariable = $("#subvariable").val();

      if (!variable || !subvariable) {
        alert(
          "Please select both a variable and a subvariable before submitting."
        );
        return false; // Prevent form submission
      }
    });
  });
});
