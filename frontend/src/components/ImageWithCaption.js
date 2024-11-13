import React from 'react';
import { Box, Typography } from '@mui/material';

const ImageWithCaption = ({ src, alt, caption }) => {
  return (
    <Box mt={4} textAlign="center">
      <img src={src} alt={alt} style={{ maxWidth: '100%', height: 'auto' }} />
      <Typography variant="caption" display="block" mt={1}>
        {caption}
      </Typography>
    </Box>
  );
};

export default ImageWithCaption;