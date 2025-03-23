## Configuration Parameters

The MODGenX system uses several configuration parameters to control model behavior:

### Unit Conversion Factors

- `fit_to_meter`: Converts feet to meters (default: 0.3048)
- `recharge_conv_factor`: Converts recharge from inch/year to meter/day (default: 0.0254/365.25)

These factors are used to ensure consistent units throughout the modeling process.

The recharge data is expected to be provided in inches/year and will be automatically converted to meters/day during processing.
