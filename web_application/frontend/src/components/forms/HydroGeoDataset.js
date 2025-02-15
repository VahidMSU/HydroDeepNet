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
    <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <FormControl fullWidth>
        <InputLabel id="variable-label" sx={{ color:"#ff8500" }}>Variable</InputLabel>
        <Select
          labelId="variable-label"
          id="variable"
          name="variable"
          value={formData.variable}
          onChange={handleChange}
        >
          <MenuItem value="">Select Variable</MenuItem>
          {availableVariables.map((variable) => (
            <MenuItem key={variable} value={variable}>
              {variable}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <FormControl fullWidth>
        <InputLabel id="subvariable-label" sx={{ color:"#ff8500" }}>Subvariable</InputLabel>
        <Select
          labelId="subvariable-label"
          id="subvariable"
          name="subvariable"
          value={formData.subvariable}
          onChange={handleChange}
        >
          <MenuItem value="">Select Subvariable</MenuItem>
          {availableSubvariables.map((subvariable) => (
            <MenuItem key={subvariable} value={subvariable}>
              {subvariable}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <Box>
        <Typography variant="subtitle1" sx={{ color:"#444e5e", fontWeight: "bold" }}>Selected Coordinates:</Typography>
        <TextField
          fullWidth
          value={formData.geometry ? 'Area selected on map' : 'Use map to select area'}
          sx={{
            input: { color: '#ff8500' }
          }}
        />
      </Box>

      <Box>
        <Typography variant="subtitle1" sx={{ color:"#444e5e", fontWeight: "bold" }}>Bounds:</Typography>
        <TextField
          fullWidth
          value={
            formData.min_latitude
              ? `Lat: ${formData.min_latitude} to ${formData.max_latitude}, Lon: ${formData.min_longitude} to ${formData.max_longitude}`
              : 'Use map to select area'
          }
          sx={{
            input: { color: '#ff8500' }
          }}
        />
      </Box>

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


// Original Backup

// import React from 'react';
// import {
//   StyledForm,
//   FormField,
//   Label,
//   Select,
//   Input,
//   ReadOnlyInput,
//   SubmitButton,
//   CoordinatesDisplay,
//   CoordinateField,
//   CoordinateLabel,
// } from '../../styles/HydroGeoDataset.tsx';

// const HydroGeoDatasetForm = ({
//   formData,
//   handleChange,
//   handleSubmit,
//   availableVariables,
//   availableSubvariables,
// }) => {
//   return (
//     <StyledForm onSubmit={handleSubmit}>
//       <FormField>
//         <Label htmlFor="variable">Variable:</Label>
//         <Select id="variable" name="variable" value={formData.variable} onChange={handleChange}>
//           <option value="">Select Variable</option>
//           {availableVariables.map((variable) => (
//             <option key={variable} value={variable}>
//               {variable}
//             </option>
//           ))}
//         </Select>
//       </FormField>

//       <FormField>
//         <Label htmlFor="subvariable">Subvariable:</Label>
//         <Select
//           id="subvariable"
//           name="subvariable"
//           value={formData.subvariable}
//           onChange={handleChange}
//         >
//           <option value="">Select Subvariable</option>
//           {availableSubvariables.map((subvariable) => (
//             <option key={subvariable} value={subvariable}>
//               {subvariable}
//             </option>
//           ))}
//         </Select>
//       </FormField>

//       <CoordinatesDisplay>
//         <CoordinateField>
//           <CoordinateLabel>Selected Coordinates:</CoordinateLabel>
//           <ReadOnlyInput
//             type="text"
//             value={formData.geometry ? 'Area selected on map' : 'No area selected'}
//             disabled
//           />
//         </CoordinateField>

//         <CoordinateField>
//           <CoordinateLabel>Bounds:</CoordinateLabel>
//           <ReadOnlyInput
//             type="text"
//             value={
//               formData.min_latitude
//                 ? `Lat: ${formData.min_latitude} to ${formData.max_latitude}, Lon: ${formData.min_longitude} to ${formData.max_longitude}`
//                 : 'Use map to select area'
//             }
//             disabled
//           />
//         </CoordinateField>
//       </CoordinatesDisplay>

//       <SubmitButton type="submit" variant="contained" disabled={!formData.geometry}>
//         Fetch Data
//       </SubmitButton>
//     </StyledForm>
//   );
// };

// export default HydroGeoDatasetForm;
