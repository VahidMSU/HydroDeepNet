// Update the MapView component to use the new Refresh Stations button
const MapControlsContainer = () => {
  const {
    showStationPanel,
    toggleStationPanel,
    mapSelectionLoading,
    refreshStationGeometries,
    handleMapRefresh,
  } = useContext(SWATGenXContext);

  return (
    <MapControlsContainer>
      <MapControlButton title="Station selection tool" className="active">
        <FontAwesomeIcon icon={faMousePointer} />
      </MapControlButton>
      <MapControlButton
        title={showStationPanel ? 'Hide station list' : 'Show station list'}
        onClick={toggleStationPanel}
        className={showStationPanel ? 'active' : ''}
      >
        <FontAwesomeIcon icon={faListUl} />
      </MapControlButton>
      <MapControlButton
        title="Refresh station data"
        onClick={refreshStationGeometries}
        disabled={mapSelectionLoading}
      >
        <FontAwesomeIcon icon={faSync} className={mapSelectionLoading ? 'fa-spin' : ''} />
      </MapControlButton>
      <MapControlButton
        title="Refresh map display"
        onClick={handleMapRefresh}
        disabled={mapSelectionLoading}
        style={{ backgroundColor: '#4a90e2' }}
      >
        <FontAwesomeIcon icon={faRedoAlt} />
      </MapControlButton>
    </MapControlsContainer>
  );
};
