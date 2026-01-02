
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

// Log startup config (masking keys)
console.log('-------------------------------------------');
console.log(' NEBULA RAG BACKEND STARTING (ESM MODE) ');
console.log(` Target Index: ${config.indexName}`);
console.log(` Supabase URL: ${config.supabaseUrl}`);
console.log('-------------------------------------------');

const ai = new GoogleGenAI({ apiKey: config.geminiKey });
const supabase = createClient(config.supabaseUrl, config.supabaseKey);

/**
 * PINECONE HELPERS
 */
async function getPineconeHost() {
  console.log(`[SYS] Resolving Pinecone Host for: ${config.indexName}`);
  const response = await fetch(`https://api.pinecone.io/indexes/${config.indexName}`, {
    headers: { "Api-Key": config.pineconeKey }
  });
  if (!response.ok) throw new Error(`Pinecone host fetch failed: ${response.statusText}`);
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
  if (!response.ok) throw new Error(`OpenAI embedding failed: ${response.statusText}`);
  const json = await response.json();
  return json.data[0].embedding;
}

/**
 * ENDPOINTS
 */

// 1. Text Extraction (OCR/Files)
app.post('/api/extract', upload.array('files'), async (req, res) => {
  console.log(`[IO] Extraction Request: ${req.files?.length || 0} files received.`);
  try {
    const extractedDocs = [];
    for (const file of req.files) {
      console.log(`[IO] Parsing: ${file.originalname}`);
      let text = "";
      if (file.mimetype.startsWith('image/')) {
        console.log(`[AI] Running Vision-OCR for ${file.originalname}`);
        const result = await ai.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: {
            parts: [
              { text: "Extract all text from this image exactly." },
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
    console.log(`[IO] Extraction Success: ${extractedDocs.length} documents ready.`);
    res.json({ docs: extractedDocs });
  } catch (e) {
    console.error(`[ERR] Extraction Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 2. Indexing (Embed + Pinecone)
app.post('/api/index', async (req, res) => {
  console.log(`[SYS] Indexing Pipeline: Processing ${req.body.docs?.length} documents.`);
  try {
    const { docs } = req.body;
    const host = await getPineconeHost();
    const vectors = [];

    for (const doc of docs) {
      console.log(`[AI] Creating embeddings for: ${doc.name}`);
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

    console.log(`[SYS] Pinecone Upsert: Shipping ${vectors.length} vectors to ${host}`);
    const upsertRes = await fetch(`https://${host}/vectors/upsert`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey },
      body: JSON.stringify({ vectors })
    });

    if (!upsertRes.ok) throw new Error(`Pinecone upsert failed: ${upsertRes.statusText}`);
    
    console.log(`[SYS] Vector Sync Complete.`);
    res.json({ success: true, count: vectors.length });
  } catch (e) {
    console.error(`[ERR] Indexing Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 3. Complexity Determination
app.post('/api/complexity', async (req, res) => {
  console.log(`[AI] Analyzing data complexity heuristics...`);
  try {
    const { docCount, charCount } = req.body;
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Perform heuristic analysis on dataset: 
      Documents: ${docCount}
      Total Characters: ${charCount}
      
      Determine if this dataset is SIMPLE or COMPLEX. 
      Rules: docCount > 3 OR charCount > 10000 = COMPLEX.
      Return the level and a concise technical reason.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            level: { type: Type.STRING, description: "SIMPLE or COMPLEX" },
            reason: { type: Type.STRING }
          },
          required: ["level", "reason"]
        }
      }
    });
    console.log(`[AI] Complexity Analysis: ${response.text}`);
    res.json(JSON.parse(response.text));
  } catch (e) {
    console.error(`[ERR] Complexity Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 4. Knowledge Graph Generation
app.post('/api/analyze_graph', async (req, res) => {
  console.log(`[AI] Mapping Knowledge Topology...`);
  try {
    const { fullText } = req.body;
    const snippet = fullText?.slice(0, 15000) || "";
    
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Parse semantic entities and relations from text: "${snippet}".
      Identify core concepts as nodes and their interactions as links.
      Rules:
      1. Nodes must have unique IDs and a group integer (1-5).
      2. Links must reference node IDs.
      3. Use technical but readable labels.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            nodes: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  label: { type: Type.STRING },
                  group: { type: Type.INTEGER }
                },
                required: ["id", "label", "group"]
              }
            },
            links: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  source: { type: Type.STRING },
                  target: { type: Type.STRING },
                  relation: { type: Type.STRING },
                  value: { type: Type.INTEGER }
                },
                required: ["source", "target", "relation", "value"]
              }
            }
          },
          required: ["nodes", "links"]
        }
      }
    });

    const graph = JSON.parse(response.text);
    console.log(`[AI] Topology Extracted: ${graph.nodes?.length} nodes, ${graph.links?.length} edges.`);
    res.json(graph);
  } catch (e) {
    console.error(`[ERR] Graph Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 5. RAG Query
app.post('/api/query', async (req, res) => {
  console.log(`[AI] RAG Inference Sequence Start.`);
  try {
    const { messages } = req.body;
    const lastUserMsg = messages[messages.length - 1].content;
    
    console.log(`[AI] Calculating query vector...`);
    const queryEmbedding = await getOpenAIEmbedding(lastUserMsg);
    const host = await getPineconeHost();

    console.log(`[SYS] Vector Retrieval: Querying ${host}`);
    const queryRes = await fetch(`https://${host}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey },
      body: JSON.stringify({ vector: queryEmbedding, topK: 5, includeMetadata: true })
    });
    const { matches } = await queryRes.json();

    const context = matches.map(m => `[File: ${m.metadata.source}]\n${m.metadata.text}`).join("\n\n---\n\n");
    console.log(`[AI] Context Injected: ${matches.length} matches found.`);
    
    const completionRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${config.openaiKey}`
      },
      body: JSON.stringify({
        model: "gpt-4-turbo-preview",
        messages: [
          { role: "system", content: `You are Nebula RAG Assistant. Answer based strictly on provided context. Cite sources using [Source Name].\n\nContext:\n${context}` },
          ...messages.map(m => ({ role: m.role === 'model' ? 'assistant' : 'user', content: m.content }))
        ]
      })
    });
    const completionJson = await completionRes.json();
    console.log(`[AI] Inference Complete.`);
    res.json({ 
      answer: completionJson.choices[0].message.content, 
      citations: matches.map(m => ({ id: m.id, metadata: { source: m.metadata.source } })) 
    });
  } catch (e) {
    console.error(`[ERR] Query Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// Start server
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`[SYS] Backend Listening on ${PORT}`);
  console.log(`[SYS] Environment Check: OK`);
});
