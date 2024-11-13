import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Typography, Box, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close'; // Import Close icon for the remove button
import ImageIcon from '@mui/icons-material/Image';

import { getCookies, acceptedImages } from '../../utils/helpers';

const ImagesDropzone = ({ images, setImages, onDrop, isLoading }) => {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedImages,
    multiple: true,
    disabled: isLoading, // Disable dropzone when loading
  });

  const [formattedFiles, setFormattedFiles] = useState([]);

  useEffect(() => {
    const formattedFiles = images.map((file, index) => ({
      id: index, // Add an id to each file
      name: file.name.substring(0, 12) + (file.name.length > 12 ? '...' : '') + ' (' + (file.size / 1024 / 1024).toFixed(2) + ' MB)',
      originalName: file.name,
      size: file.size
    }));
    setFormattedFiles(formattedFiles);
  }, [images]);

  const removeFile = (id) => {
    const newImages = images.filter((_, index) => index !== id);
    setImages(newImages);
  };

  return (
    <>
      <Typography variant="h6">Select images to process:</Typography>
      <Box {...getRootProps()} sx={{ border: '2px dashed #ccc', padding: 2, textAlign: 'center', cursor: isLoading ? 'not-allowed' : 'pointer' }}>
        <input {...getInputProps()} disabled={isLoading} />
        {isDragActive ? (
          <Typography>Drop the image files here ...</Typography>
        ) : (
          <>
            <Typography>Drag 'n' drop image files here, or click to select image files</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', mt: 1 }}>
              {formattedFiles.map((file) => (
                <Box key={file.id} sx={{
                  mr: 1, mt: 1, padding: '2px 8px', border: '1px solid #ccc', borderRadius: '4px',
                  backgroundColor: '#1E2A45', color: '#fff', display: 'flex', alignItems: 'center'
                }}>
                  <ImageIcon fontSize="small" sx={{ mr: 0.5 }} />{file.name}
                  <IconButton onClick={() => removeFile(file.id)} size="small" sx={{ ml: 1 }} disabled={isLoading}>
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </Box>
              ))}
            </Box>
          </>
        )}
      </Box>
    </>
  );
};

export default ImagesDropzone;