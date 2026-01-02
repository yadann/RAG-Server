// server.js (ES Module version)

import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { GoogleGenAI } from '@google/genai';
import { createClient } from '@supabase/supabase-js';

const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));

const app = express();
const upload = multer();

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Configuration from Environment Variables
const config = {
  openaiKey: process.env.OPENAI_API_KEY,
  pineconeKey: process.env.PINECONE_API,
  geminiKey: process.env.GOOGLE_GENAI_KEY,
  supabaseUrl: process.env.SUPABASE_URL,
  supabaseKey: process.env.SUPABASE_KEY,
  indexName: process.env.INDEX_NAME || 'clean-user'
};

const ai = new GoogleGenAI({ apiKey: config.geminiKey });
const supabase = createClient(config.supabaseUrl, config.supabaseKey);

/**
 * PINECONE HELPERS
 */
async function getPineconeHost() {
  const response = await fetch(`https://api.pinecone.io/indexes/${config.indexName}`, {
    headers: { "Api-Key": config.pineconeKey }
  });
  const json = await response.json();
  return json.host;
}

async function getOpenAIEmbedding(text) {
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${config.openaiKey}`
    },
    body: JSON.stringify({
      input: text,
      model: "text-embedding-3-small",
      dimensions: 1024
    })
  });
  const json = await response.json();
  return json.data[0].embedding;
}

/**
 * ENDPOINTS
 */

// 1. Text Extraction (OCR/Files)
app.post('/api/extract', upload.array('files'), async (req, res) => {
  try {
    const extractedDocs = [];

    for (const file of req.files) {
      let text = "";

      if (file.mimetype.startsWith('image/')) {
        const model = ai.getGenerativeModel({ model: "gemini-1.5-flash" });
        const result = await model.generateContent([
          "Extract all text from this image exactly.",
          { inlineData: { data: file.buffer.toString('base64'), mimeType: file.mimetype } }
        ]);
        text = result.response.text();
      } else {
        text = file.buffer.toString('utf-8');
      }

      extractedDocs.push({
        id: Math.random().toString(36).substring(7),
        name: file.originalname,
        type: file.mimetype,
        content: text,
        size: file.size
      });
    }

    res.json({ docs: extractedDocs });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// 2. Indexing (Embed + Pinecone)
app.post('/api/index', async (req, res) => {
  try {
    const { docs } = req.body;
    const host = await getPineconeHost();
    const vectors = [];

    for (const doc of docs) {
      const chunks = doc.content.match(/[\s\S]{1,2000}/g) || [doc.content];
      for (let i = 0; i < chunks.length; i++) {
        const embedding = await getOpenAIEmbedding(chunks[i]);
        vectors.push({
          id: `${doc.id}-${i}`,
          values: embedding,
          metadata: { text: chunks[i], source: doc.name, docId: doc.id }
        });
      }
    }

    // Batch Upsert
    await fetch(`https://${host}/vectors/upsert`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey },
      body: JSON.stringify({ vectors })
    });

    res.json({ success: true, count: vectors.length });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// 3. RAG Query
app.post('/api/query', async (req, res) => {
  try {
    const { messages } = req.body;
    const lastUserMsg = messages[messages.length - 1].content;
    const queryEmbedding = await getOpenAIEmbedding(lastUserMsg);
    const host = await getPineconeHost();

    const queryRes = await fetch(`https://${host}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey },
      body: JSON.stringify({ vector: queryEmbedding, topK: 5, includeMetadata: true })
    });

    const { matches } = await queryRes.json();
    const context = matches.map(m => `[File: ${m.metadata.source}]\n${m.metadata.text}`).join("\n\n---\n\n");

    const completionRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${config.openaiKey}`
      },
      body: JSON.stringify({
        model: "gpt-4-turbo-preview",
        messages: [
          { role: "system", content: `You are a helpful assistant. Use context:\n\n${context}` },
          ...messages.map(m => ({ role: m.role === 'model' ? 'assistant' : 'user', content: m.content }))
        ]
      })
    });

    const completionJson = await completionRes.json();
    res.json({ 
      answer: completionJson.choices[0].message.content, 
      citations: matches.map(m => ({ id: m.id, metadata: { source: m.metadata.source } }))
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Start server
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
