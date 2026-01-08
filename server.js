import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { GoogleGenAI, Type } from '@google/genai';
import { createClient } from '@supabase/supabase-js';
import fetch from 'node-fetch';
import crypto from 'crypto';

const app = express();
const upload = multer();

app.use(cors());
app.use(express.json({ limit: '100mb' }));

const config = {
  openaiKey: process.env.OPENAI_API_KEY,
  pineconeKey: process.env.PINECONE_API,
  geminiKey: process.env.GOOGLE_GENAI_KEY,
  supabaseUrl: process.env.SUPABASE_URL,
  supabaseKey: process.env.SUPABASE_KEY,
  indexName: process.env.INDEX_NAME || 'clean-user'
};

const ai = new GoogleGenAI({ apiKey: config.geminiKey });

/**
 * CORE HELPERS
 */
const getHash = (text) => crypto.createHash('sha256').update(text).digest('hex');

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

app.post('/api/extract', upload.array('files'), async (req, res) => {
  try {
    const extractedDocs = [];
    for (const file of req.files) {
      let text = "";
      if (file.mimetype.startsWith('image/') || file.mimetype === 'application/pdf') {
        const result = await ai.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: {
            parts: [
              { text: "Extract text accurately." },
              {
                inlineData: {
                  data: file.buffer.toString('base64'),
                  mimeType: file.mimetype
                }
              }
            ]
          }
        });
        text = result.text;
      } else {
        text = file.buffer.toString('utf-8');
      }
      extractedDocs.push({
        id: getHash(file.originalname).slice(0, 12),
        name: file.originalname,
        content: text,
        type: file.mimetype,
        size: file.size
      });
    }
    res.json({ docs: extractedDocs });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Optimized Indexing with Stable IDs and Namespace Isolation
app.post('/api/index', async (req, res) => {
  try {
    const { docs, username } = req.body;
    const host = await getPineconeHost();
    const namespace = `user-${username}`;
    const vectors = [];

    for (const doc of docs) {
      const chunks = doc.content.match(/[\s\S]{1,1500}/g) || [doc.content];
      const fileHash = getHash(doc.name).slice(0, 8);

      const BATCH_SIZE = 10;
      for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
        const currentBatch = chunks.slice(i, i + BATCH_SIZE);
        const embeddings = await Promise.all(
          currentBatch.map(chunk => getOpenAIEmbedding(chunk))
        );

        embeddings.forEach((emb, index) => {
          const contentHash = getHash(currentBatch[index]).slice(0, 16);
          vectors.push({
            id: `vec_${fileHash}_${contentHash}`,
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

    const P_BATCH = 100;
    for (let i = 0; i < vectors.length; i += P_BATCH) {
      await fetch(`https://${host}/vectors/upsert`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Api-Key": config.pineconeKey
        },
        body: JSON.stringify({
          vectors: vectors.slice(i, i + P_BATCH),
          namespace
        })
      });
    }

    res.json({ success: true, count: vectors.length });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Enhanced Graph Generation with Community Summarization
app.post('/api/analyze_graph', async (req, res) => {
  try {
    const { fullText, username } = req.body;
    const host = await getPineconeHost();
    const communityNamespace = `community-summaries-user-${username}`;

    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Analyze this text and generate a hierarchical Knowledge Graph. 
Identify entities (nodes) and relations (links). 
Crucially: Group entities into 'communities' (logical clusters) and provide a 200-word summary for each cluster.
Text: "${fullText.slice(0, 15000)}"`,
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
                  group: { type: Type.INTEGER },
                  summary: { type: Type.STRING }
                },
                required: ["id", "label", "group", "summary"]
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

    const communityVectors = [];
    for (const node of graph.nodes) {
      if (node.summary) {
        const emb = await getOpenAIEmbedding(node.summary);
        communityVectors.push({
          id: `comm_${getHash(node.id).slice(0, 16)}`,
          values: emb,
          metadata: {
            text: node.summary,
            community_id: node.id,
            type: 'community_summary'
          }
        });
      }
    }

    if (communityVectors.length > 0) {
      await fetch(`https://${host}/vectors/upsert`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Api-Key": config.pineconeKey
        },
        body: JSON.stringify({
          vectors: communityVectors,
          namespace: communityNamespace
        })
      });
    }

    res.json(graph);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Stage 1 & 2 RAG: Resolver + Answerer
app.post('/api/query', async (req, res) => {
  try {
    const { messages, username, ragMode } = req.body;
    const lastUserMsg = messages[messages.length - 1].content;
    const host = await getPineconeHost();

    // --- STAGE 1: RESOLVER ---
    const resolverRes = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Act as a RAG Controller. Decide if this query needs context or history only.
Rules: If query contains technical entities or facts, context needed. If it's a greeting or pronoun follow-up, history only.
Return JSON: { "use_context": boolean, "resolved_task": "de-contextualized query" }
History: ${JSON.stringify(messages.slice(-3))}
Current Query: ${lastUserMsg}`,
      config: { responseMimeType: "application/json" }
    });

    const plan = JSON.parse(resolverRes.text);

    let context = "";
    let matches = [];

    if (plan.use_context) {
      const queryEmbedding = await getOpenAIEmbedding(plan.resolved_task);

      const isGraph = ragMode === 'GRAPH' ||
        (ragMode === 'AUTO' && messages.length > 5);

      const targetNamespace = isGraph
        ? `community-summaries-user-${username}`
        : `user-${username}`;

      const queryRes = await fetch(`https://${host}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Api-Key": config.pineconeKey
        },
        body: JSON.stringify({
          vector: queryEmbedding,
          topK: 5,
          includeMetadata: true,
          namespace: targetNamespace
        })
      });

      const queryData = await queryRes.json();
      matches = queryData.matches || [];
      context = matches.map(m =>
        `[Source: ${m.metadata.source || 'Community Summary'}]\n${m.metadata.text}`
      ).join("\n\n---\n\n");
    }

    // --- STAGE 2: ANSWERER (GPT-5 NANO) ---
    const completionRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${config.openaiKey}`
      },
      body: JSON.stringify({
        model: "gpt-5-nano",
        messages: [
          {
            role: "system",
            content: `You are Nebula Assistant. Cite sources as [Source Name].\n\nContext:\n${context}`
          },
          ...messages.map(m => ({
            role: m.role === 'model' ? 'assistant' : 'user',
            content: m.content
          }))
        ]
      })
    });

    const completionJson = await completionRes.json();

    res.json({
      answer: completionJson.choices[0].message.content,
      citations: matches.map(m => ({
        id: m.id,
        metadata: m.metadata
      }))
    });

  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`[SYS] Server Active on ${PORT}`));
