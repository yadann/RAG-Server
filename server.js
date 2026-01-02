
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { GoogleGenAI, Type } from '@google/genai';
import { createClient } from '@supabase/supabase-js';
import fetch from 'node-fetch';

const app = express();
const upload = multer();

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); // Increased limit for large documents

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
  if (!response.ok) throw new Error(`Pinecone unreachable: ${response.statusText}`);
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
  if (!response.ok) throw new Error(`OpenAI Embedding Error: ${response.statusText}`);
  const json = await response.json();
  return json.data[0].embedding;
}

/**
 * ENDPOINTS
 */

// 1. Enhanced Extraction (OCR/PDF/Text)
app.post('/api/extract', upload.array('files'), async (req, res) => {
  console.log(`[IO] Extraction Request: ${req.files?.length} files`);
  try {
    const extractedDocs = [];
    for (const file of req.files) {
      console.log(`[IO] Processing: ${file.originalname} (${file.mimetype})`);
      let text = "";

      // Use Gemini for vision (images) and PDF extraction
      if (file.mimetype.startsWith('image/') || file.mimetype === 'application/pdf') {
        console.log(`[AI] Running Multi-modal extraction for ${file.originalname}`);
        const result = await ai.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: {
            parts: [
              { text: "Extract and summarize all readable text from this document accurately. Preserve structure." },
              { inlineData: { data: file.buffer.toString('base64'), mimeType: file.mimetype } }
            ]
          }
        });
        text = result.text;
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
    console.error(`[ERR] Extraction:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 2. Optimized Indexing (Parallel Embeddings)
app.post('/api/index', async (req, res) => {
  console.log(`[SYS] Indexing Phase Start.`);
  try {
    const { docs } = req.body;
    const host = await getPineconeHost();
    const vectors = [];

    for (const doc of docs) {
      console.log(`[SYS] Processing Document: ${doc.name}`);
      // Use a smaller chunk size for better granularity (1000 chars)
      const chunks = doc.content.match(/[\s\S]{1,1500}/g) || [doc.content];
      console.log(`[SYS] Generated ${chunks.length} chunks for ${doc.name}`);

      // Parallelize embedding generation in small batches of 10 to avoid OpenAI rate limits
      const BATCH_SIZE = 10;
      for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
        const currentBatch = chunks.slice(i, i + BATCH_SIZE);
        console.log(`[SYS] Embedding progress: ${Math.round((i / chunks.length) * 100)}%`);
        
        const embeddingPromises = currentBatch.map(chunk => getOpenAIEmbedding(chunk));
        const embeddings = await Promise.all(embeddingPromises);

        embeddings.forEach((emb, index) => {
          vectors.push({
            id: `${doc.id}-${i + index}`,
            values: emb,
            metadata: { 
              text: currentBatch[index], 
              source: doc.name, 
              docId: doc.id 
            }
          });
        });
      }
    }

    // Pinecone batch upsert (limit 100 vectors per call)
    console.log(`[SYS] Shipping ${vectors.length} vectors to Pinecone...`);
    const P_BATCH = 100;
    for (let i = 0; i < vectors.length; i += P_BATCH) {
      const batch = vectors.slice(i, i + P_BATCH);
      const upsertRes = await fetch(`https://${host}/vectors/upsert`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey },
        body: JSON.stringify({ vectors: batch })
      });
      if (!upsertRes.ok) throw new Error(`Pinecone batch ${i} failed`);
    }

    console.log(`[SYS] Vector Index Updated.`);
    res.json({ success: true, count: vectors.length });
  } catch (e) {
    console.error(`[ERR] Indexing:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 3. Complexity Heuristics
app.post('/api/complexity', async (req, res) => {
  try {
    const { docCount, charCount } = req.body;
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Evaluate RAG dataset complexity. Docs: ${docCount}, Chars: ${charCount}. 
      Threshold: >10,000 chars is COMPLEX. Return SIMPLE/COMPLEX and technical reason.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            level: { type: Type.STRING },
            reason: { type: Type.STRING }
          },
          required: ["level", "reason"]
        }
      }
    });
    res.json(JSON.parse(response.text));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// 4. Graph Mapping
app.post('/api/analyze_graph', async (req, res) => {
  try {
    const { fullText } = req.body;
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Extract entities and relations from: "${fullText.slice(0, 12000)}". Format as JSON nodes/links.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            nodes: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { id: {type: Type.STRING}, label: {type: Type.STRING}, group: {type: Type.INTEGER} }, required: ["id", "label", "group"] } },
            links: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { source: {type: Type.STRING}, target: {type: Type.STRING}, relation: {type: Type.STRING}, value: {type: Type.INTEGER} }, required: ["source", "target", "relation", "value"] } }
          },
          required: ["nodes", "links"]
        }
      }
    });
    res.json(JSON.parse(response.text));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// 5. RAG Inference
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
    const context = matches.map(m => `[Source: ${m.metadata.source}]\n${m.metadata.text}`).join("\n\n---\n\n");

    const completionRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${config.openaiKey}` },
      body: JSON.stringify({
        model: "gpt-4-turbo-preview",
        messages: [
          { role: "system", content: `You are Nebula Assistant. Answer using ONLY this context:\n\n${context}` },
          ...messages.map(m => ({ role: m.role === 'model' ? 'assistant' : 'user', content: m.content }))
        ]
      })
    });
    const completionJson = await completionRes.json();
    res.json({ answer: completionJson.choices[0].message.content, citations: matches.map(m => ({ id: m.id, metadata: { source: m.metadata.source } })) });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`[SYS] ESM Server Active on ${PORT}`));
