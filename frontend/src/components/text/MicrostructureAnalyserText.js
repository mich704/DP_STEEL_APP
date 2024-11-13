// src/components/AnalyserParameters.js
import React from 'react';
import { Typography, Box } from '@mui/material';

const MicrostructureAnalyserText = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Microstructure Analyser
      </Typography>
      <Typography variant="body1">
            Provide images classified as dualphase steel or images that you acknowledge as dp steel.<br></br>
            Analyser will return .zip package with raport about given microstructures together with images with detailed phases of dp steel.
      </Typography>
      <br></br>
      <Typography variant="body1" paragraph>
        The analyser will return the following parameters in the report:
      </Typography>
      <Typography variant="h5" gutterBottom>
        Mean E (GPa)
      </Typography>
      <Typography variant="body1" paragraph>
        This parameter represents the mean Young's modulus (E) in gigapascals (GPa). Young's modulus is a measure of the stiffness of a material. It describes the relationship between stress (force per unit area) and strain (proportional deformation) in a material.
      </Typography>
      <Typography variant="h5" gutterBottom>
        Mean σY (MPa)
      </Typography>
      <Typography variant="body1" paragraph>
        This parameter represents the mean yield stress (σY) in megapascals (MPa). Yield stress is the stress at which a material begins to deform plastically. Before the yield point, the material will deform elastically and will return to its original shape when the applied stress is removed.
      </Typography>
      <Typography variant="h5" gutterBottom>
        Mean UTS (GPa)
      </Typography>
      <Typography variant="body1" paragraph>
        This parameter represents the mean ultimate tensile strength (UTS) in gigapascals (GPa). Ultimate tensile strength is the maximum stress that a material can withstand while being stretched or pulled before breaking.
      </Typography>
    </Box>
  );
};

export default MicrostructureAnalyserText;