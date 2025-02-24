import React from 'react';
import { Box, FormControl, InputLabel, Select, MenuItem, TextField, Button, Typography } from '@mui/material';

const HydroGeoDatasetForm = ({
  formData,
  handleChange,
  handleSubmit,
  availableVariables,
  availableSubvariables,
}) => {
  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      {/* Variable Field */}
      <FormControl fullWidth>
        <InputLabel
          id="variable-label"
          sx={{
            color: "#ff8500",
            bgcolor: 'white',
            px: 1,
            borderRadius: 1,
            '&.MuiInputLabel-shrink': {
              bgcolor: 'white',
              px: 1,
              borderRadius: 1,
            }
          }}
        >
          Variable
        </InputLabel>
        <Select
          labelId="variable-label"
          id="variable"
          name="variable"
          value={formData.variable}
          onChange={handleChange}
          sx={{
            bgcolor: 'white',
            borderRadius: 1,
          }}
        >
          <MenuItem value="">Select Variable</MenuItem>
          {availableVariables.map((variable) => (
            <MenuItem key={variable} value={variable}>
              {variable}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Subvariable Field */}
      <FormControl fullWidth>
        <InputLabel
          id="subvariable-label"
          sx={{
            color: "#ff8500",
            bgcolor: 'white',
            px: 1,
            borderRadius: 1,
            '&.MuiInputLabel-shrink': {
              bgcolor: 'white',
              px: 1,
              borderRadius: 1,
            }
          }}
        >
          Subvariable
        </InputLabel>
        <Select
          labelId="subvariable-label"
          id="subvariable"
          name="subvariable"
          value={formData.subvariable}
          onChange={handleChange}
          sx={{
            bgcolor: 'white',
            borderRadius: 1,
          }}
        >
          <MenuItem value="">Select Subvariable</MenuItem>
          {availableSubvariables.map((subvariable) => (
            <MenuItem key={subvariable} value={subvariable}>
              {subvariable}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Selected Coordinates */}
      <Box>
        <Typography variant="subtitle1" sx={{ color: "#444e5e", fontWeight: "bold" }}>Selected Coordinates:</Typography>
        <TextField
          fullWidth
          value={formData.geometry ? 'Area selected on map' : 'Use map to select area'}
          sx={{
            input: { color: '#ff8500' },
            bgcolor: 'white',
            borderRadius: 1,
          }}
        />
      </Box>

      {/* Bounds */}
      <Box>
        <Typography variant="subtitle1" sx={{ color: "#444e5e", fontWeight: "bold" }}>Bounds:</Typography>
        <TextField
          fullWidth
          value={
            formData.min_latitude
              ? `Lat: ${formData.min_latitude} to ${formData.max_latitude}, Lon: ${formData.min_longitude} to ${formData.max_longitude}`
              : 'Use map to select area'
          }
          sx={{
            input: { color: '#ff8500' },
            bgcolor: 'white',
            borderRadius: 1,
          }}
        />
      </Box>

      {/* Fetch Data Button */}
      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          type="submit"
          variant="contained"
          disabled={!formData.geometry}
          sx={{
            backgroundColor: '#ff8500',
            fontWeight: 'bold',
            width: 'fit-content',
            paddingX: 3,
            '&:hover': {
              backgroundColor: '#e67500',
            }
          }}
        >
          Fetch Data
        </Button>
      </Box>
    </Box>
  );
};

export default HydroGeoDatasetForm;
