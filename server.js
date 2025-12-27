const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(cors({ origin: '*' }));
app.use(bodyParser.json({ limit: '50mb' }));

// Health Check
app.get('/', (req, res) => {
  res.status(200).json({ status: 'active', version: '1.4.0-stable' });
});

app.post('/', async (req, res) => {
  const { operation, messages } = req.body;
  try {
    if (!process.env.API_KEY) throw new Error("API_KEY_NOT_FOUND");
    
    // 1. New Client Style
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

    if (operation === "QUERY") {
      const lastMessage = messages[messages.length - 1].content;
      
      // 2. Direct Call (No getGenerativeModel)
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: lastMessage,
      });

      // 3. Property Access (No .text())
      return res.json({ answer: response.text, citations: [] });
    }
    
    res.json({ status: 'success' });
  } catch (error) {
    console.error("Critical Failure:", error.message);
    res.status(500).json({ error: error.message });
  }
});

app.listen(process.env.PORT || 3000, () => console.log("Nebula Ready"));
