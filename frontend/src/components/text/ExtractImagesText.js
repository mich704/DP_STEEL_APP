import React from 'react';
import { Typography, Box } from '@mui/material';

const ExtractImagesText = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Images Extractor
      </Typography>
      <Typography variant="body1" gutterBottom>
        This tool allows you to provide scientific publications in PDF format to extract images from them. 
        It is designed to help researchers and scientists easily obtain images embedded within their publications 
        for further analysis and classification.
      </Typography>
      <Typography variant="body1" gutterBottom>
        You can also extract images and classify them based on the selected classification type. 
        <br/>
        The classification types available are:
      </Typography>
      <ul>
        <li>
          <Typography variant="body1">
            <strong>Image Classification</strong> - 
            This option classifies images into two categories: <em>microstructure</em> and <em>rest</em>. 
            It helps in identifying and segregating images 
            that contain microstructural details from other types of images.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>Microstructure Classification</strong> - This option further classifies images containing 
            microstructures into two categories: <em>dualphase</em> and <em>rest</em>. It is useful for 
            distinguishing between dual-phase microstructures and other types of microstructures.
          </Typography>
        </li>
      </ul>
    </Box>
  );
};

export default ExtractImagesText;