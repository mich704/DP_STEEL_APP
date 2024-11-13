import React from 'react';
import { CircularProgress, Box, Typography } from '@mui/material';

const LoadingOverlay = ({ message }) => {
  return (
    <Box
      sx={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        width: '50%',
        height: '50%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.5)', // Increased opacity
        zIndex: 1000,
        borderRadius: '12px', // Add border radius for smoother corners
        transform: 'translate(-50%, -50%)', // Center the overlay
      }}
    >
      <CircularProgress />
      {message && (
        <Typography variant="h6" sx={{ marginTop: 2, color: 'white' }}>
          {message}
        </Typography>
      )}
    </Box>
  );
};

export default LoadingOverlay;