import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Button, FormControl, InputLabel, MenuItem, Select, Typography, Box } from '@mui/material';
import { v4 as uuidv4 } from 'uuid';

import DownloadImagesBtn from './DownloadImagesBtn';
import { getCookie, fetchGetRequestFromApi } from '../utils/helpers'; // Import the getCookie function
import PublicationsDropzone from './dropzones/PublicationsDropzone'; // Import the new component
import LoadingOverlay from './LoadingOverlay'; // Import the LoadingOverlay component
import ExtractImagesText from './text/ExtractImagesText';

function ExtractImagesForm() {
  const [publications, setPublications] = useState([]);
  const [classificationType, setClassificationType] = useState('none');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [isResponseOk, setIsResponseOk] = useState(false);
  const [isButtonDisabled, setIsButtonDisabled] = useState(false);
  const [isLoading, setIsLoading] = useState(false); // Add loading state
  const [loadingMessage, setLoadingMessage] = useState(''); // Add loading message state
  const [sessionId, setSessionId] = useState(null);
  const [triggerWebSocket, setTriggerWebSocket] = useState(false);

  const wsRef = useRef(null);
  const intervalRef = useRef(null);
  const socketRef = useRef(null);

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
    const fetchData = async () => {
      setLoadingMessage('Fetching initial data...');
      await fetchGetRequestFromApi('http://127.0.0.1:8000/api/extract_images/');
      setLoadingMessage('');
    };
    fetchData();
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
    wsRef.current = new WebSocket(`ws://localhost:8000/ws/extract_images/${sessionId}/`);
  
    wsRef.current.onopen = () => {
      console.log('WebSocket connection established');
  
      //Send a message every 2 seconds
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

  const classificationTypes = [
    { value: 'none', label: 'None (just extract images)' },
    { value: 'image', label: 'Image Classification' },
    { value: 'microstructure', label: 'Microstructure Classification' },
  ];

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    // Filter accepted PDF files
    const pdfFiles = acceptedFiles.filter(file => file.type === 'application/pdf');
    setPublications(prevPublications => {
      const newFiles = pdfFiles.filter(file => 
        !prevPublications.some(pub => pub.name === file.name)
      );
      return [...prevPublications, ...newFiles];
    });

    // Display a warning for rejected files
    if (rejectedFiles.length > 0) {
      alert('Only PDF files are allowed. Please drop only PDF files.');
      console.log('Rejected files:', rejectedFiles);
    }
  }, []);

  const handleClassificationChange = (event) => {
    setClassificationType(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsButtonDisabled(true);
    setIsLoading(true); // Set loading state to true
    setLoadingMessage('Processing your request...'); // Set loading message
    if (!publications.length) {
      alert('Please select a file before submitting.');
      setIsButtonDisabled(false);
      setIsLoading(false); // Set loading state to false
      setLoadingMessage(''); // Clear loading message
      return;
    }

    const formData = new FormData();
    publications.forEach((pub) => {
      formData.append('publications', pub);
    });
    formData.append('classification_type', classificationType);
    const csrftoken = getCookie('csrftoken');

    if (!isButtonDisabled){
      //setIsButtonDisabled(false);
      setDownloadUrl('');
    }

    try {
      const response = await fetch('/api/extract_images/', {
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
      console.log(jsonResponse);
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
      setPublications([]);
    }
  }, [isButtonDisabled, isResponseOk, downloadUrl]);

  return (
    <>
      <ExtractImagesText/>
      <Box component="form" onSubmit={handleSubmit} 
          sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxWidth: 500, margin: 'auto', position: 'relative' }}>
        <Box sx={{ position: 'relative' }}>
          {isLoading && <LoadingOverlay message={loadingMessage} />} {/* Conditionally render LoadingOverlay with message */}
          <PublicationsDropzone 
              publications={publications} 
              setPublications={setPublications} 
              onDrop={onDrop} 
              isLoading={isLoading} 
          />
        </Box>
        
        <FormControl fullWidth>
          <InputLabel>Select classification type</InputLabel>
          <Select
            value={classificationType}
            label="Select classification type"
            onChange={handleClassificationChange}
            required
          >
            {classificationTypes.map((type) => (
              <MenuItem key={type.value} value={type.value}>
                {type.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Button type="submit" variant="contained" disabled={isButtonDisabled}>Process</Button>
        {isResponseOk && downloadUrl !== '' && 
            <div ref={btnRef}>
              <DownloadImagesBtn download_url={downloadUrl} />
            </div>
        }
        
      </Box>
    </>
    
  );
}

export default ExtractImagesForm;