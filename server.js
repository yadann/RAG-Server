const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

// 1. Health Check (Prevents Render.com deployment timeouts)
app.get('/', (req, res) => res.status(200).send('Nebula Server is Online'));

// 2. Main Gateway (Handles both root / and /ingest paths)
app.post('/', (req, res) => {
  const { operation, docs, messages } = req.body;
  console.log(`[Gateway] Received ${operation} request`);

  if (operation === "INGEST") {
    // Return simulated success
    return res.json({ 
      status: 'success', 
      processed_chunks: (docs?.length || 0) * 5 
    });
  }

  if (operation === "QUERY") {
    return res.json({ 
      answer: "Connection successful! This response came from your Render server.", 
      citations: [] 
    });
  }

  res.status(400).json({ error: "Unknown operation" });
});

// Alias for legacy paths
app.post('/ingest', (req, res) => res.redirect(307, '/'));
app.post('/query', (req, res) => res.redirect(307, '/'));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server Live on port ${PORT}`));
