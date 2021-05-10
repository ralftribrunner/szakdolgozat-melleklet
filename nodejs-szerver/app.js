// Létrehoz egy szervert, mely a localhost 8080-as portján egy Hello World-el válaszol
'use strict';

const express = require('express');

// Constants
const PORT = 8080;
const HOST = '0.0.0.0';

// App
const app = express();
app.get('/', (req, res) => {
  res.send('Hello World');
});

app.listen(PORT, HOST); // szerver elindítása
console.log(`Running on http://${HOST}:${PORT}`);

// forrás:
// https://www.digitalocean.com/community/tutorials/nodejs-express-basics