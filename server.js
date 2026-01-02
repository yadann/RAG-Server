
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { GoogleGenAI, Type } = require('@google/genai');
const { createClient } = require('@supabase/supabase-js');
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

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
  console.log(`[Pinecone] Fetching host for index: ${config.indexName}`);
  const response = await fetch(`https://api.pinecone.io/indexes/${config.indexName}`, {
    headers: { "Api-Key": config.pineconeKey }
  });
  if (!response.ok) throw new Error(`Pinecone host fetch failed: ${response.statusText}`);
  const json = await response.json();
  console.log(`[Pinecone] Host resolved: ${json.host}`);
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
  console.log(`[API/Extract] Received ${req.files?.length} files`);
  try {
    const extractedDocs = [];
    for (const file of req.files) {
      console.log(`[API/Extract] Processing: ${file.originalname} (${file.mimetype})`);
      let text = "";
      if (file.mimetype.startsWith('image/')) {
        console.log(`[API/Extract] Running OCR via Gemini for ${file.originalname}`);
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
    console.log(`[API/Extract] Success. Extracted ${extractedDocs.length} docs.`);
    res.json({ docs: extractedDocs });
  } catch (e) {
    console.error(`[API/Extract] Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 2. Indexing (Embed + Pinecone)
app.post('/api/index', async (req, res) => {
  console.log(`[API/Index] Starting indexing for ${req.body.docs?.length} docs`);
  try {
    const { docs } = req.body;
    const host = await getPineconeHost();
    const vectors = [];

    for (const doc of docs) {
      console.log(`[API/Index] Chunking and embedding: ${doc.name}`);
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

    console.log(`[API/Index] Upserting ${vectors.length} vectors to Pinecone...`);
    const upsertRes = await fetch(`https://${host}/vectors/upsert`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey },
      body: JSON.stringify({ vectors })
    });

    if (!upsertRes.ok) throw new Error(`Pinecone upsert failed: ${upsertRes.statusText}`);
    
    console.log(`[API/Index] Indexing complete.`);
    res.json({ success: true, count: vectors.length });
  } catch (e) {
    console.error(`[API/Index] Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 3. Complexity Determination
app.post('/api/complexity', async (req, res) => {
  console.log(`[API/Complexity] Analyzing dataset heuristics...`);
  try {
    const { docCount, charCount } = req.body;
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Analyze the following data statistics and return a complexity level (SIMPLE or COMPLEX) and a reason.
      Docs: ${docCount}, Total Chars: ${charCount}.
      If docCount > 3 or charCount > 10000, mark as COMPLEX.`,
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
    console.log(`[API/Complexity] Heuristics result: ${response.text}`);
    res.json(JSON.parse(response.text));
  } catch (e) {
    console.error(`[API/Complexity] Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 4. Knowledge Graph Generation
app.post('/api/analyze_graph', async (req, res) => {
  console.log(`[API/AnalyzeGraph] Generating knowledge topology...`);
  try {
    const { fullText } = req.body;
    const snippet = fullText?.slice(0, 15000) || "";
    
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Extract a semantic knowledge graph from this text: "${snippet}". 
      Identify entities and relationships. Format as JSON with 'nodes' and 'links'.
      Nodes need 'id', 'label', and 'group' (integer).
      Links need 'source' (node id), 'target' (node id), 'relation', and 'value' (integer).`,
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

    console.log(`[API/AnalyzeGraph] Graph generated with ${JSON.parse(response.text).nodes?.length} nodes.`);
    res.json(JSON.parse(response.text));
  } catch (e) {
    console.error(`[API/AnalyzeGraph] Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// 5. RAG Query
app.post('/api/query', async (req, res) => {
  console.log(`[API/Query] Handling RAG request...`);
  try {
    const { messages } = req.body;
    const lastUserMsg = messages[messages.length - 1].content;
    
    console.log(`[API/Query] Embedding user query...`);
    const queryEmbedding = await getOpenAIEmbedding(lastUserMsg);
    const host = await getPineconeHost();

    console.log(`[API/Query] Searching vector store...`);
    const queryRes = await fetch(`https://${host}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey },
      body: JSON.stringify({ vector: queryEmbedding, topK: 5, includeMetadata: true })
    });
    const { matches } = await queryRes.json();

    const context = matches.map(m => `[File: ${m.metadata.source}]\n${m.metadata.text}`).join("\n\n---\n\n");
    console.log(`[API/Query] Retrieved ${matches.length} context fragments.`);
    
    const completionRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${config.openaiKey}`
      },
      body: JSON.stringify({
        model: "gpt-4-turbo-preview",
        messages: [
          { role: "system", content: `You are a helpful assistant. Use context provided below. Be precise. If info is missing, say so.\n\nContext:\n${context}` },
          ...messages.map(m => ({ role: m.role === 'model' ? 'assistant' : 'user', content: m.content }))
        ]
      })
    });
    const completionJson = await completionRes.json();
    console.log(`[API/Query] LLM Response generated.`);
    res.json({ 
      answer: completionJson.choices[0].message.content, 
      citations: matches.map(m => ({ id: m.id, metadata: { source: m.metadata.source } })) 
    });
  } catch (e) {
    console.error(`[API/Query] Error:`, e);
    res.status(500).json({ error: e.message });
  }
});

// Start server
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`-------------------------------------------`);
  console.log(` NEBULA RAG BACKEND ACTIVE ON PORT ${PORT} `);
  console.log(`-------------------------------------------`);
});
