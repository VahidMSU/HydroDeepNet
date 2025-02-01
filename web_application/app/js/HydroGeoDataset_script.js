require([
  "esri/Map",
  "esri/views/MapView",
  "esri/layers/GraphicsLayer",
  "esri/widgets/Sketch",
  "esri/geometry/support/webMercatorUtils",
], function (Map, MapView, GraphicsLayer, Sketch, webMercatorUtils) {
  const MapApp = {
    graphicsLayer: new GraphicsLayer(),
    init() {
      this.map = new Map({
        basemap: "topo-vector",
        layers: [this.graphicsLayer],
        attribution: {
          content:
            'Data sources: <a href="https://www.usgs.gov/core-science-systems/ngp/national-hydrography" target="_blank">USGS National Hydrography Dataset</a>, <a href="https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697" target="_blank">USDA Soil Survey Geographic Database</a>',
        },
      });

      this.view = new MapView({
        container: "viewDiv",
        map: this.map,
        center: [-84.5, 43.4],
        zoom: 7,
        ui: { components: [] },
      });

      this.sketch = new Sketch({
        layer: this.graphicsLayer,
        view: this.view,
        creationMode: "update",
        visibleElements: {
          createTools: {
            point: true,
            polyline: false,
            polygon: true,
            rectangle: true,
            circle: false,
          },
          selectionTools: { "rectangle-selection": true },
        },
      });

      this.view.ui.add(this.sketch, "bottom-left");

      this.addEventListeners();
    },
    addEventListeners() {
      this.sketch.on("create", (event) => this.handleSketchEvent(event));
      this.sketch.on("update", (event) => this.handleSketchEvent(event));
      this.view.on("click", (event) => this.handleMapClick(event));

      $(document).ready(() => {
        $("#variable").on("change", () => this.handleVariableChange());
        $("form").on("submit", (e) => this.handleFormSubmit(e));
      });

      window.resetAoiFields = () => this.resetAoiFields();
    },
    handleMapClick(event) {
      const geographicPoint = webMercatorUtils.webMercatorToGeographic(
        event.mapPoint
      );
      this.updatePointFields(
        geographicPoint.latitude.toFixed(6),
        geographicPoint.longitude.toFixed(6)
      );
    },
    handleSketchEvent(event) {
      // Check if the event is in the "complete" stage
      if (event.state === "complete") {
        const geometry = event.graphic?.geometry;
        if (!geometry) {
          console.warn("No geometry found in event:", event);
          return;
        }

        const geographicGeometry =
          webMercatorUtils.webMercatorToGeographic(geometry);

        switch (geometry.type) {
          case "extent":
            this.updateExtentFields(geographicGeometry);
            break;
          case "polygon":
            this.updatePolygonFields(geographicGeometry);
            break;
          default:
            console.warn("Unsupported geometry type:", geometry.type);
        }
      }
    },
    updatePointFields(lat, lon) {
      document.querySelector("#latitude").value = lat;
      document.querySelector("#longitude").value = lon;
      console.log("Point fields updated:", lat, lon);
    },
    updateExtentFields(extent) {
      document.querySelector('[name="min_latitude"]').value =
        extent.ymin.toFixed(6);
      document.querySelector('[name="max_latitude"]').value =
        extent.ymax.toFixed(6);
      document.querySelector('[name="min_longitude"]').value =
        extent.xmin.toFixed(6);
      document.querySelector('[name="max_longitude"]').value =
        extent.xmax.toFixed(6);
      console.log("Extent fields updated:", extent);
    },
    updatePolygonFields(polygon) {
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
      console.log("Polygon fields updated:", {
        min_latitude: Math.min(...latitudes).toFixed(6),
        max_latitude: Math.max(...latitudes).toFixed(6),
      });
    },
    resetAoiFields() {
      document.querySelector("#polygon_coordinates").value = "";
      document.querySelector('[name="min_latitude"]').value = "";
      document.querySelector('[name="max_latitude"]').value = "";
      document.querySelector('[name="min_longitude"]').value = "";
      document.querySelector('[name="max_longitude"]').value = "";
      document.querySelector("#latitude").value = "";
      document.querySelector("#longitude").value = "";
      console.log("AOI fields reset.");
    },
    handleVariableChange() {
      const selectedVariable = $("#variable").val();
      const subvariableDropdown = $("#subvariable").empty();
      if (selectedVariable) {
        const csrfToken = $('input[name="csrf_token"]').val();
        this.fetchSubvariables(
          selectedVariable,
          csrfToken,
          subvariableDropdown
        );
      } else {
        subvariableDropdown.append(
          '<option value="">Select a variable first</option>'
        );
      }
    },
    handleFormSubmit(e) {
      const variable = $("#variable").val();
      const subvariable = $("#subvariable").val();
      if (!variable || !subvariable) {
        e.preventDefault();
        alert(
          "Please select both a variable and a subvariable before submitting."
        );
        console.warn(
          "Form submission blocked: variable or subvariable not selected."
        );
      }
    },
    fetchSubvariables(variable, csrfToken, dropdown) {
      $.ajax({
        url: "/get_subvariables",
        method: "POST",
        headers: { "X-CSRFToken": csrfToken },
        data: { variable },
        success: (response) => {
          const subvariables = response.subvariables || [];
          dropdown.empty();
          if (subvariables.length === 0) {
            dropdown.append(
              '<option value="">No subvariables available</option>'
            );
          } else {
            dropdown.append(
              subvariables
                .map((sub) => `<option value="${sub}">${sub}</option>`)
                .join("")
            );
          }
        },
        error: (xhr, status, error) => {
          console.error("Error fetching subvariables:", error);
          alert("Failed to load subvariables. Please try again.");
        },
      });
    },
  };

  // Initialize the app
  MapApp.init();
});
