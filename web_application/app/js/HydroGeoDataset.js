require([
  "esri/Map",
  "esri/views/MapView",
  "esri/layers/GraphicsLayer",
  "esri/widgets/Sketch",
  "esri/geometry/support/webMercatorUtils",
], (Map, MapView, GraphicsLayer, Sketch, webMercatorUtils) => {
  const MapApp = {
    initialView: { center: [-84.5, 43.4], zoom: 7 },

    init() {
      // Initialize map, layer and view.
      this.graphicsLayer = new GraphicsLayer();
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
        center: this.initialView.center,
        zoom: this.initialView.zoom,
        ui: { components: [] },
      });

      // Initialize Sketch widget.
      this.sketch = new Sketch({
        layer: this.graphicsLayer,
        view: this.view,
        creationMode: "update",
        visibleElements: {
          createTools: { point: true, polygon: true, rectangle: true },
          selectionTools: { "rectangle-selection": true },
        },
      });
      this.view.ui.add(this.sketch, "bottom-left");

      this.registerEventListeners();
    },

    registerEventListeners() {
      // Listen to Sketch events.
      this.sketch.on("create", (evt) => this.handleSketchEvent(evt));
      this.sketch.on("update", (evt) => this.handleSketchEvent(evt));
      // Map click to update point fields.
      this.view.on("click", (evt) => this.handleMapClick(evt));

      // Register DOM events only after ensuring the document is ready.
      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () =>
          this.setupDomListeners()
        );
      } else {
        this.setupDomListeners();
      }
    },

    setupDomListeners() {
      const variableEl = document.querySelector("#variable");
      if (variableEl) {
        variableEl.addEventListener("change", () =>
          this.handleVariableChange()
        );
      } else {
        console.warn("Element with id #variable not found.");
      }

      const formEl = document.querySelector("form");
      if (formEl) {
        formEl.addEventListener("submit", (e) => this.handleFormSubmit(e));
      } else {
        console.warn("Form element not found.");
      }

      const resetBtn = document.querySelector("#resetButton");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => this.resetApplication());
      }
    },

    handleMapClick(event) {
      const geoPoint = webMercatorUtils.webMercatorToGeographic(event.mapPoint);
      this.updatePointFields(
        geoPoint.latitude.toFixed(6),
        geoPoint.longitude.toFixed(6)
      );
    },

    handleSketchEvent(event) {
      const { graphic } = event;
      if (!graphic || !graphic.geometry) {
        console.warn("No graphic/geometry in event:", event);
        return;
      }
      const { geometry } = graphic;
      const geoGeom = webMercatorUtils.webMercatorToGeographic(geometry);

      if (geometry.type === "polygon") {
        // Enforce single polygon.
        if (event.state === "start") {
          this.ensureSinglePolygon(graphic);
        }
        if (event.state === "active" || event.state === "complete") {
          this.updatePolygonFields(geoGeom);
        }
      } else if (geometry.type === "extent" && event.state === "complete") {
        this.updateExtentFields(geoGeom);
      } else if (geometry.type === "point" && event.state === "complete") {
        this.updatePointFields(
          geoGeom.latitude.toFixed(6),
          geoGeom.longitude.toFixed(6)
        );
      }
    },

    ensureSinglePolygon(currentGraphic) {
      // Remove any polygon with a different uid.
      const toRemove = this.graphicsLayer.graphics.filter(
        (g) => g.geometry.type === "polygon" && g.uid !== currentGraphic.uid
      );
      toRemove.forEach((g) => this.graphicsLayer.remove(g));
    },

    updatePointFields(lat, lon) {
      const latInput = document.querySelector("#latitude");
      const lonInput = document.querySelector("#longitude");
      if (latInput) {
        latInput.value = lat;
      }
      if (lonInput) {
        lonInput.value = lon;
      }
      console.log("Point fields updated:", lat, lon);
    },

    updateExtentFields(extent) {
      const minLatEl = document.querySelector('[name="min_latitude"]');
      const maxLatEl = document.querySelector('[name="max_latitude"]');
      const minLonEl = document.querySelector('[name="min_longitude"]');
      const maxLonEl = document.querySelector('[name="max_longitude"]');
      if (minLatEl) {
        minLatEl.value = extent.ymin.toFixed(6);
      }
      if (maxLatEl) {
        maxLatEl.value = extent.ymax.toFixed(6);
      }
      if (minLonEl) {
        minLonEl.value = extent.xmin.toFixed(6);
      }
      if (maxLonEl) {
        maxLonEl.value = extent.xmax.toFixed(6);
      }
      console.log("Extent fields updated:", extent);
    },

    updatePolygonFields(polygon) {
      const coordinates = (polygon.rings && polygon.rings[0]) || [];
      if (!coordinates.length) {
        console.warn("No coordinates in polygon.");
        return;
      }
      const latitudes = coordinates.map((coord) => coord[1]);
      const longitudes = coordinates.map((coord) => coord[0]);
      const minLat = Math.min(...latitudes).toFixed(6);
      const maxLat = Math.max(...latitudes).toFixed(6);
      const minLon = Math.min(...longitudes).toFixed(6);
      const maxLon = Math.max(...longitudes).toFixed(6);

      const minLatEl = document.querySelector('[name="min_latitude"]');
      const maxLatEl = document.querySelector('[name="max_latitude"]');
      const minLonEl = document.querySelector('[name="min_longitude"]');
      const maxLonEl = document.querySelector('[name="max_longitude"]');
      const polyCoordsEl = document.querySelector("#polygon_coordinates");

      if (minLatEl) {
        minLatEl.value = minLat;
      }
      if (maxLatEl) {
        maxLatEl.value = maxLat;
      }
      if (minLonEl) {
        minLonEl.value = minLon;
      }
      if (maxLonEl) {
        maxLonEl.value = maxLon;
      }
      if (polyCoordsEl) {
        polyCoordsEl.value = JSON.stringify(coordinates);
      }

      console.log("Polygon fields updated:", {
        minLat,
        maxLat,
        minLon,
        maxLon,
      });
    },

    resetAoiFields() {
      [
        "#polygon_coordinates",
        '[name="min_latitude"]',
        '[name="max_latitude"]',
        '[name="min_longitude"]',
        '[name="max_longitude"]',
        "#latitude",
        "#longitude",
      ].forEach((sel) => {
        const el = document.querySelector(sel);
        if (el) {
          el.value = "";
        }
      });
      console.log("AOI fields reset.");
    },

    async handleVariableChange() {
      const variableEl = document.querySelector("#variable");
      const subvariableEl = document.querySelector("#subvariable");
      if (!variableEl || !subvariableEl) {
        console.warn("Variable or subvariable element not found.");
        return;
      }
      // Clear existing options.
      subvariableEl.innerHTML = "";
      const selectedVariable = variableEl.value;
      if (!selectedVariable) {
        subvariableEl.innerHTML =
          '<option value="">Select a variable first</option>';
        return;
      }
      try {
        const csrfToken =
          document.querySelector('input[name="csrf_token"]')?.value || "";
        const subvariables = await this.fetchSubvariables(
          selectedVariable,
          csrfToken
        );
        if (subvariables.length) {
          subvariableEl.innerHTML = subvariables
            .map((sub) => `<option value="${sub}">${sub}</option>`)
            .join("");
        } else {
          subvariableEl.innerHTML =
            '<option value="">No subvariables available</option>';
        }
      } catch (error) {
        console.error("Error fetching subvariables:", error);
        alert("Failed to load subvariables. Please try again.");
      }
    },

    handleFormSubmit(e) {
      const variableEl = document.querySelector("#variable");
      const subvariableEl = document.querySelector("#subvariable");
      if (
        !variableEl ||
        !subvariableEl ||
        !variableEl.value ||
        !subvariableEl.value
      ) {
        e.preventDefault();
        alert(
          "Please select both a variable and a subvariable before submitting."
        );
        console.warn(
          "Form submission blocked: variable or subvariable not selected."
        );
      }
    },

    async fetchSubvariables(variable, csrfToken) {
      // Use URLSearchParams to send data as form-encoded.
      const params = new URLSearchParams();
      params.append("variable", variable);
      const response = await fetch("/get_subvariables", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
          "X-CSRFToken": csrfToken,
        },
        body: params,
      });
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      const data = await response.json();
      return data.subvariables || [];
    },

    resetApplication() {
      this.graphicsLayer.removeAll();
      if (this.sketch) {
        this.sketch.cancel();
      }
      this.resetAoiFields();
      if (this.view) {
        this.view.goTo({
          center: this.initialView.center,
          zoom: this.initialView.zoom,
        });
      }
      console.log("Application reset.");
    },
  };

  // Initialize the application.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => MapApp.init());
  } else {
    MapApp.init();
  }
});
