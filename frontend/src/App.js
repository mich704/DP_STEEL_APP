import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Container, Box, Fab, CssBaseline } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import Brightness4Icon from '@mui/icons-material/Brightness4';

import ExtractImagesForm from './components/ExtractImagesForm';
import MicrostructureAnalyserForm from './components/MicrostructureAnalyserForm';
import HomePageText from './components/text/HomePageText';

function App() {
  const lightTheme = createTheme({
    palette: {
      mode: 'light',
    },
  });

  const darkTheme = createTheme({
    palette: {
      mode: 'dark',
    },
  });

  const [theme, setTheme] = useState(darkTheme);
  const toggleTheme = () => {
    setTheme(theme.palette.mode === 'light' ? darkTheme : lightTheme);
  };

  useEffect(() => {
    const checkLiveness = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/liveness/');
        const data = await response.json();
        console.log('Liveness status:', data.status);
      } catch (error) {
        console.log('Not live');
      }
    };
    checkLiveness();
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              DualPhase Steel Analyser
            </Typography>
            <Button color="inherit" component={Link} to="/">Home</Button>
            <Button color="inherit" component={Link} to="/extract-images">Extract Images</Button>
            <Button color="inherit" component={Link} to="/microstructure-analyser">Analyser</Button>
          </Toolbar>
        </AppBar>
        <Container>
          <Box my={4}>
            <Routes>
              <Route path="/" element={<HomePageText/>} />
              <Route path="/extract-images" element={<ExtractImagesForm/>} />
              <Route path="/microstructure-analyser" element={<MicrostructureAnalyserForm/>} />
              {/* Add other routes here */}
            </Routes>
         
          </Box>
        </Container>
       
      </Router>
    </ThemeProvider>
  );
}

export default App;