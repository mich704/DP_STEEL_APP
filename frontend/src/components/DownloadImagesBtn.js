import React, { useState } from 'react';
import Button from '@mui/material/Button';
import DownloadIcon from '@mui/icons-material/Download';
import '../styles/DownloadImagesBtn.css';

function DownloadImagesBtn({ download_url }) {
  const [downloadUrl, setDownloadUrl] = useState(download_url);

  // Function to refresh the page
  const refreshPage = () => {
    window.location.reload();
  };

  return (
    <div className="centered-container">
      {downloadUrl && (
        <Button
          variant="contained"
          color="primary"
          href={downloadUrl}
          download
          onClick={refreshPage}
          startIcon={<DownloadIcon />}
        >
          Download Results
        </Button>
      )}
    </div>
  );
}

export default DownloadImagesBtn;