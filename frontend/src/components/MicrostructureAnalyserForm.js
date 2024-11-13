import React, { useEffect, useState, useRef, useCallback, Suspense } from 'react';
import { Button, Typography, Box } from '@mui/material';
import { v4 as uuidv4 } from 'uuid';

import { getCookie, acceptedImages, fetchGetRequestFromApi } from '../utils/helpers';
import ImagesDropzone from './dropzones/ImagesDropzone';
import DownloadImagesBtn from './DownloadImagesBtn';
import MicrostructureAnalyserText from './text/MicrostructureAnalyserText';
import LoadingOverlay from './LoadingOverlay';

function MicrostructureAnalyserForm() {
  const [images, setImages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState('');
  const [isButtonDisabled, setIsButtonDisabled] = useState(false);
  const [isResponseOk, setIsResponseOk] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [triggerWebSocket, setTriggerWebSocket] = useState(false);
  
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  // Function to generate and store a unique session ID
  const createSession = () => {
    const newSessionId = uuidv4();
    sessionStorage.setItem('sessionId', newSessionId);
    setSessionId(newSessionId);
  };

  const clearSession = () => {
    sessionStorage.removeItem('sessionId');
    setSessionId(null);
  };

  useEffect(() => {
    const storedSessionId = sessionStorage.getItem('sessionId');
    if (!storedSessionId) {
      createSession();
    } else {
      setSessionId(storedSessionId);
    }
  }, []);

  useEffect(() => {
    // Establish WebSocket connection
    const sessionId = sessionStorage.getItem('sessionId');
    wsRef.current = new WebSocket(`ws://localhost:8000/ws/microstructure_analyser/${sessionId}/`);
  
    wsRef.current.onopen = () => {
      console.log('WebSocket connection established');
  
      // Send a message every 2 seconds
      // intervalRef.current = setInterval(() => {
      //   wsRef.current.send(JSON.stringify({ message: 'Periodic message' }));
      // }, 2000);
    };
  
    wsRef.current.onmessage = (event) => {
      const loadingMessage = JSON.parse(event.data);
      console.log('WebSocket message received:', loadingMessage);
      setLoadingMessage(loadingMessage.message);
      // Handle incoming WebSocket messages here
    };
  
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  
    return () => {
      // Clean up WebSocket connection and interval
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [triggerWebSocket]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsButtonDisabled(true);
    setIsLoading(true);
    setLoadingMessage('Processing your request...');
    if (!images.length) {
      alert('Please provide images to analyse.');
      setIsButtonDisabled(false);
      setIsLoading(false);
      setLoadingMessage('');
      return;
    }
    const formData = new FormData();
    images.forEach((image) => {
      formData.append('images', image);
    });
    const csrftoken = getCookie('csrftoken');

    if (!isButtonDisabled){
      setDownloadUrl('');
    }

    try {
      const response = await fetch('/api/microstructure_analysis/', {
        method: 'POST',
        body: formData,
        headers: {
          'X-CSRFToken': csrftoken,
          'session-id': sessionId,
        },
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const jsonResponse = await response.json();
      setDownloadUrl(jsonResponse.download_url);
      setIsResponseOk(true);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      clearSession(); // Clear the session after the request ends
      setIsButtonDisabled(false);
      setIsLoading(false); // Set loading state to false
      setLoadingMessage(''); // Clear loading message
      setTriggerWebSocket(prev => !prev);
    }
  };

  const btnRef = useRef(null);

  useEffect(() => {
    if (isResponseOk && downloadUrl !== '') {
      btnRef.current.scrollIntoView({ behavior: 'smooth' });
      setImages([]);
    }
  }, [isResponseOk, downloadUrl]);

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    // Filter accepted image files
    const imageFiles = acceptedFiles.filter(file => {
      const fileType = file.type;
      return Object.keys(acceptedImages).some(type => fileType === type);
    });

    setImages(prevImages => {
      const newFiles = imageFiles.filter(file =>
        !prevImages.some(pub => pub.name === file.name)
      );
      return [...prevImages, ...newFiles];
    });

    // Display a warning for rejected files
    if (rejectedFiles.length > 0) {
      alert('Only image files are allowed.');
      console.log('Rejected files:', rejectedFiles);
    }
  }, []);

  return (
    <>
      <MicrostructureAnalyserText />
      <Box component="form" onSubmit={handleSubmit}
        sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxWidth: 500, margin: 'auto', position: 'relative' }}>
        <Box sx={{ position: 'relative' }}>
          {isLoading && <LoadingOverlay message={loadingMessage}/>}
          <ImagesDropzone images={images} setImages={setImages} onDrop={onDrop} />
        </Box>
        <Button type="submit" variant="contained" disabled={isButtonDisabled}>Analyse microstructures</Button>
        {isResponseOk && downloadUrl !== '' &&
          <div ref={btnRef}>
            <DownloadImagesBtn
              download_url={downloadUrl}
            />
          </div>
        }
      </Box>
    </>
  );
}

export default MicrostructureAnalyserForm;