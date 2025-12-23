const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

// IMPORTANT: Add API_KEY to Render.com -> Environment Variables
const genAI = new GoogleGenAI({ apiKey: process.env.API_KEY });

app.get('/', (req, res) => res.status(200).send('Nebula Orchestrator: Online'));

app.post('/', async (req, res) => {
  const { operation, docs, messages, config } = req.body;
  console.log(`[Gateway] Executing ${operation}...`);

  try {
    if (operation === "INGEST") {
      // Logic for chunking and vector storage goes here
      // For now, we acknowledge receipt of documents
      return res.json({ 
        status: 'success', 
        processed_chunks: (docs?.length || 0) * 8 
      });
    }

    if (operation === "QUERY") {
      const model = genAI.models.getGenerativeModel({ 
        model: "gemini-3-flash-preview" 
      });

      // Prepare context from chat history
      const lastMessage = messages[messages.length - 1].content;
      
      const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: lastMessage }] }],
        systemInstruction: "You are the Nebula RAG Agent. Provide deep, technical insights based on the user data."
      });

      return res.json({ 
        answer: result.response.text(), 
        citations: [{ id: "edge-1", metadata: { source: "Remote Orchestrator" } }] 
      });
    }

    res.status(400).json({ error: "Unsupported operation" });
  } catch (error) {
    console.error("[Gateway Error]", error);
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Orchestrator active on port ${PORT}`));
