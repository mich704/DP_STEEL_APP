import React from 'react';
import { Typography, Box } from '@mui/material';
import suspension from '../../images/suspension.png'; 

import ImageWithCaption from '../ImageWithCaption';

const HomePageText = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dual-Phase Steel
      </Typography>
      <Typography variant="body1" paragraph>
        Dual-phase (DP) steel is a type of high-strength steel that is characterized by a microstructure consisting of a soft ferrite phase and a hard martensite phase. This unique combination provides an excellent balance of strength and ductility, making DP steel highly desirable in various applications, particularly in the automotive industry.
      </Typography>
      <Typography variant="h5" gutterBottom>
        Key Features:
      </Typography>
      <ul>
        <li>
          <Typography variant="body1">
            <strong>High Strength:</strong> The presence of martensite increases the tensile strength of the steel.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>Good Ductility:</strong> The ferrite phase ensures that the steel remains ductile and formable.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>Enhanced Formability:</strong> DP steel can be easily formed into complex shapes, which is essential for manufacturing automotive components.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>Crashworthiness:</strong> The high energy absorption capacity of DP steel improves vehicle safety during collisions.
          </Typography>
        </li>
      </ul>
      <Typography variant="h5" gutterBottom>
        Applications:
      </Typography>
      <ul>
        <li>
          <Typography variant="body1">
            <strong>Automotive Industry:</strong> Used in the manufacturing of car bodies, structural components, and safety parts.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>Construction:</strong> Employed in building structures that require a combination of strength and flexibility.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>Machinery:</strong> Utilized in the production of high-strength machine parts.
          </Typography>
        </li>
      </ul>
      <Typography variant="body1" paragraph>
        Dual-phase steel represents a significant advancement in materials engineering, offering a versatile solution for industries that demand both high performance and reliability.
      </Typography>
      <ImageWithCaption
        src={suspension}
        alt="Zawieszenie"
        caption="Image: Suspension system of a car, showcasing the application of dual-phase steel."
      />
    </Box>
  );
};

export default HomePageText;