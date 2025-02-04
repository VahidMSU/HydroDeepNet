function selectStation(stationNumber) {
  const userInput = document.getElementById("user_input");
  userInput.value = stationNumber;
  // Force the 'change' event so geometry fetch code is triggered immediately
  userInput.dispatchEvent(new Event("change"));
}

// Toggles visibility of calibration, sensitivity, or validation panels
document.getElementById("viewDiv").style.height = "900px";

function toggleSettings(setting) {
  const settingsContainer = document.getElementById(`${setting}_settings`);
  const checkbox = document.getElementById(`${setting}_flag`);
  settingsContainer.style.display = checkbox.checked ? "block" : "none";
}

// JavaScript for Esri Map and Station Details
require([
  "esri/Map",
  "esri/views/MapView",
  "esri/Graphic",
  "esri/layers/GraphicsLayer",
  "esri/geometry/Polygon",
  "esri/geometry/Polyline",
  "esri/widgets/LayerList",
  "esri/widgets/BasemapToggle",
  "esri/widgets/Search",
], function (
  Map,
  MapView,
  Graphic,
  GraphicsLayer,
  Polygon,
  Polyline,
  LayerList,
  BasemapToggle,
  Search
) {
  // 1) Initialize map & view
  const map = new Map({ basemap: "topo-vector" });
  const view = new MapView({
    container: "viewDiv",
    map: map,
    zoom: 4,
    center: [-90, 38],
  });
  // 2) Create separate layers for polygons and streams
  const polygonLayer = new GraphicsLayer({ title: "HUC12 Polygons" });
  const streamLayer = new GraphicsLayer({ title: "Stream Polylines" });
  const lakeLayer = new GraphicsLayer({ title: "Lake Polygons" });
  // Optional: a separate marker layer if you want that toggled independently
  // const stationMarkerLayer = new GraphicsLayer({ title: "Station Marker" });

  // Add them to the map so each appears in the LayerList
  map.addMany([polygonLayer, streamLayer, lakeLayer]);
  // map.add(stationMarkerLayer); // if using a marker layer

  // 3) Add layer controls
  const layerList = new LayerList({ view: view });
  view.ui.add(layerList, "top-right");

  const basemapToggle = new BasemapToggle({
    view: view,
    nextBasemap: "satellite",
  });
  view.ui.add(basemapToggle, "bottom-left");

  const searchWidget = new Search({ view: view });
  view.ui.add(searchWidget, "top-left");

  // =========== (A) SEARCH BY SITE NAME ===========
  const searchButton = document.getElementById("search_button");
  searchButton.addEventListener("click", async function () {
    const searchTerm = document.getElementById("search_input").value.trim();
    showLoading(); // Show the loading indicator
    try {
      const response = await fetch(`/search_site?search_term=${searchTerm}`);
      const data = await response.json();

      const resultsDiv = document.getElementById("search_results");
      if (data.error) {
        resultsDiv.innerHTML = `<p>${data.error}</p>`;
      } else {
        // Show a clickable list of results
        let html = "<ul>";
        data.forEach((site) => {
          html += `
              <li style="cursor:pointer; color:blue"
                  onclick="selectStation('${site.SiteNumber}')">
                <strong>${site.SiteName}</strong> (Number: ${site.SiteNumber})
              </li>
            `;
        });
        html += "</ul>";
        resultsDiv.innerHTML = html;
      }
    } catch (error) {
      console.error("Error fetching search results:", error);
    } finally {
      hideLoading(); // Hide indicator when done (success or fail)
    }
  });
  // =========== (B) FETCH GEOMETRIES & CHARACTERISTICS BY STATION NUMBER ===========
  document
    .getElementById("user_input")
    .addEventListener("change", async function () {
      const selectedStation = this.value;
      showLoading(); // Show the loading indicator
      try {
        const response = await fetch(
          `/get_station_characteristics?station=${selectedStation}`
        );
        const data = await response.json();

        // Build station characteristics HTML
        let characteristicsHtml = "<h5>Station Characteristics:</h5><ul>";
        for (const key in data) {
          if (
            data.hasOwnProperty(key) &&
            key !== "geometries" &&
            key !== "streams_geometries" &&
            key !== "lakes_geometries" &&
            data[key]
          ) {
            characteristicsHtml += `<li>${key}: ${data[key]}</li>`;
          }
        }
        characteristicsHtml += "</ul>";
        document.getElementById("station_characteristics").innerHTML =
          characteristicsHtml;

        // Clear previous shapes
        polygonLayer.removeAll();
        streamLayer.removeAll();
        lakeLayer.removeAll();
        // If you have a station marker layer, also do: stationMarkerLayer.removeAll();

        // If lat/long exist, place a marker in view.graphics or a marker layer
        if (data.Latitude && data.Longitude) {
          // Zoom to station
          view.goTo({ center: [data.Longitude, data.Latitude], zoom: 10 });

          // Create a station marker
          const marker = new Graphic({
            geometry: {
              type: "point",
              longitude: data.Longitude,
              latitude: data.Latitude,
            },
            symbol: {
              type: "simple-marker",
              color: [226, 119, 40],
              outline: { color: [255, 255, 255], width: 2 },
            },
          });
          // Option 1: add it to the view's default graphics
          view.graphics.removeAll();
          view.graphics.add(marker);

          // Option 2: or put it on a separate layer so user can toggle it
          // stationMarkerLayer.add(marker);
        }

        // Draw polygons (HUC12 geometries) on polygonLayer
        if (data.geometries) {
          data.geometries.forEach((geom) => {
            const polygon = new Polygon({ rings: geom.coordinates[0] });
            const polygonGraphic = new Graphic({
              geometry: polygon,
              symbol: {
                type: "simple-fill",
                color: [227, 139, 79, 0.8], // fill color
                outline: { color: [255, 255, 255], width: 1 },
              },
            });
            polygonLayer.add(polygonGraphic);
          });
        }
        // Draw polygons (lake geometries) on lakeLayer
        if (data.lakes_geometries) {
          data.lakes_geometries.forEach((lake) => {
            const polygon = new Polygon({ rings: lake.coordinates[0] });
            const polygonGraphic = new Graphic({
              geometry: polygon,
              symbol: {
                type: "simple-fill",
                color: [0, 0, 255, 0.5], // fill color
                outline: { color: [255, 255, 255], width: 1 },
              },
            });
            lakeLayer.add(polygonGraphic);
          });
        }
        // Draw polylines (stream geometries) on streamLayer
        if (data.streams_geometries) {
          data.streams_geometries.forEach((stream) => {
            const polyline = new Polyline({ paths: stream.coordinates });
            const polylineGraphic = new Graphic({
              geometry: polyline,
              symbol: {
                type: "simple-line",
                color: [0, 0, 255],
                width: 0.5,
              },
            });
            streamLayer.add(polylineGraphic);
          });
        }
      } catch (error) {
        console.error("Error fetching station details:", error);
      } finally {
        hideLoading(); // Hide indicator when done (success or fail)
      }
    });

  // =========== (C) HANDLE RUN BUTTON ===========
  document
    .getElementById("submit_button")
    .addEventListener("click", function () {
      if (confirm("Are you sure you want to submit these settings?")) {
        document.getElementById("model_settings_form").submit();
      }
    });
});
function showLoading() {
  const loadingDiv = document.getElementById("loadingIndicator");
  if (loadingDiv) {
    loadingDiv.style.display = "block";
  }
}

function hideLoading() {
  const loadingDiv = document.getElementById("loadingIndicator");
  if (loadingDiv) {
    loadingDiv.style.display = "none";
  }
}
