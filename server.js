
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

// --- DYNAMIC AI CLIENT ---
// Instead of a global instance, we create one per request to support BYOK (Bring Your Own Key).
const getAI = (req) => {
    // Check custom header first, then fall back to env var
    const apiKey = req.headers['x-gemini-api-key'] || config.geminiKey;
    
    if (!apiKey) {
        throw new Error("No Gemini API Key provided. Please configure GOOGLE_GENAI_KEY on server or provide 'x-gemini-api-key' header.");
    }
    
    return new GoogleGenAI({ apiKey });
};

/**
 * CORE HELPERS
 */
const getHash = (text) => crypto.createHash('sha256').update(text).digest('hex');

async function getPineconeHost() {
  const response = await fetch(`https://api.pinecone.io/indexes/${config.indexName}`, {
    headers: { "Api-Key": config.pineconeKey || '' }
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
 * Safely parses JSON from an LLM's output, handling markdown wrappers and embedded objects.
 */
function parseLlmJson(text) {
    if (!text) return {};
    try {
        // Handle markdown code blocks ```json ... ```
        const cleanedText = text.replace(/^```json\s*/, '').replace(/```$/, '').trim();
        return JSON.parse(cleanedText);
    } catch (e1) {
        // Handle cases where JSON is just embedded in text
        try {
            const match = text.match(/\{[\s\S]*\}/);
            if (match && match[0]) {
                return JSON.parse(match[0]);
            }
        } catch (e2) {
            console.error("Failed to parse LLM JSON after multiple attempts:", text, e2);
        }
    }
    console.warn("Could not parse valid JSON from LLM output:", text);
    return {}; // Return empty on error to prevent a server crash
}


/**
 * ENDPOINTS
 */

app.post('/api/extract', upload.array('files'), async (req, res) => {
  try {
    const ai = getAI(req); // Initialize with request-specific key
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
        text = result.text || "";
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

// Optimized Indexing with Stable IDs and PROJECT Isolation
app.post('/api/index', async (req, res) => {
  try {
    const { docs, username, projectId } = req.body;
    
    if (!projectId) throw new Error("Project ID is required for indexing.");

    const host = await getPineconeHost();
    // Namespace format: u_{username}_p_{projectId}
    const namespace = `u_${username}_p_${projectId}`;
    const vectors = [];

    for (const doc of docs) {
      const chunks = doc.content.match(/[\s\S]{1,1500}/g) || [doc.content];
      const fileHash = getHash(doc.name).slice(0, 8);

      const BATCH_SIZE = 10;
      for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
        const currentBatch = chunks.slice(i, i + BATCH_SIZE);
        const embeddings = await Promise.all(
          currentBatch.map((chunk) => getOpenAIEmbedding(chunk))
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
          "Api-Key": config.pineconeKey || ''
        },
        body: JSON.stringify({
          vectors: vectors.slice(i, i + P_BATCH),
          namespace
        })
      });
    }

    res.json({ success: true, count: vectors.length, namespace });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Enhanced Graph Generation with Community Summarization & Project Isolation
app.post('/api/analyze_graph', async (req, res) => {
  console.log('[API] /api/analyze_graph called.');
  try {
    const ai = getAI(req); // Initialize with request-specific key
    const { fullText, username, projectId } = req.body;

    if (!fullText || !username || !projectId) {
        console.error('ANALYZE_GRAPH_ERROR: Missing required parameters.');
        return res.status(400).json({ error: "Missing fullText, username, or projectId." });
    }
    console.log(`[API] Analyzing text for project ${projectId}. Text length: ${fullText.length}`);

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

    console.log('[API] Raw Gemini response for graph analysis received.');
    const rawText = response.text || '{}';
    let graph;
    try {
        graph = JSON.parse(rawText);
    } catch (parseError) {
        console.error('ANALYZE_GRAPH_ERROR: Failed to parse JSON from Gemini response.');
        console.error('Raw Text:', rawText);
        console.error('Parse Error:', parseError);
        // Return an empty graph to prevent client crash
        return res.json({ nodes: [], links: [] }); 
    }
    console.log(`[API] Graph parsed successfully. Nodes: ${graph.nodes?.length || 0}, Links: ${graph.links?.length || 0}`);


    const communityVectors = [];
    if (graph.nodes) {
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
    }

    if (communityVectors.length > 0) {
      await fetch(`https://${host}/vectors/upsert`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Api-Key": config.pineconeKey || ''
        },
        body: JSON.stringify({
          vectors: communityVectors,
          namespace: `summary_u_${username}_p_${projectId}`
        })
      });
    }

    res.json(graph);
  } catch (e) {
    console.error('ANALYZE_GRAPH_ERROR: An unexpected error occurred in the endpoint.');
    console.error(e); // Log the full error object
    res.status(500).json({ error: e.message, trace: e.stack });
  }
});

// Stage 1 & 2 RAG: Resolver + Answerer with Project Context
app.post('/api/query', async (req, res) => {
  console.log('[API] /api/query called.');
  try {
    const ai = getAI(req); // Initialize with request-specific key for Resolver
    const { messages, username, projectId, ragMode, useWebSearch } = req.body;
    
    if (!projectId) throw new Error("Project ID is required for querying.");

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

    const plan = parseLlmJson(resolverRes.text || '');

    let context = "";
    let matches = [];

    // Force context if Web Search is on (Hybrid RAG) or if Resolver says so
    if (plan.use_context || useWebSearch) {
      const queryText = plan.resolved_task || lastUserMsg;
      const queryEmbedding = await getOpenAIEmbedding(queryText);
      const isGraph = ragMode === 'complex';
      const targetNamespace = isGraph
        ? `summary_u_${username}_p_${projectId}`
        : `u_${username}_p_${projectId}`;

      const queryRes = await fetch(`https://${host}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey || '' },
        body: JSON.stringify({
          vector: queryEmbedding,
          topK: 5,
          includeMetadata: true,
          namespace: targetNamespace
        })
      });

      const queryData = await queryRes.json();
      matches = queryData.matches || [];
      context = matches.map((m) =>
        `[Source: ${m.metadata.source || 'Community Summary'}]\n${m.metadata.text}`
      ).join("\n\n---\n\n");
    }

    if (useWebSearch) {
      console.log('[API] Web search path initiated.');
      const systemPrompt = `You are Nebula Assistant. 
You have access to Google Search to answer the user's question with real-time information.
You also have access to the user's uploaded documents via the context below.
Combine both sources to provide a comprehensive answer.
Always cite your sources.

User Documents Context:
${context || "No relevant local documents found."}`;
      
      console.log(`[API] System prompt constructed. Length: ${systemPrompt.length}`);
      
      const history = messages.slice(0, -1);
      const current_user_message = messages[messages.length - 1];

      const contentsForGemini = [
        ...history.map(m => ({
          role: m.role === 'model' ? 'model' : 'user',
          parts: [{ text: m.content }]
        })),
        {
          role: 'user',
          parts: [{ text: `${systemPrompt}\n\n${current_user_message.content}` }]
        }
      ];

      console.log('[API] Final payload for Gemini Web Search:');
      console.log(JSON.stringify(contentsForGemini, null, 2));

      const googleRes = await ai.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: contentsForGemini,
        config: {
          tools: [{ googleSearch: {} }],
        }
      });
      console.log('[API] Received response from Gemini Web Search.');

      const webCitations = googleRes.candidates?.[0]?.groundingMetadata?.groundingChunks
        ?.map((c) => c.web?.uri)
        .filter(Boolean) || [];
      
      const allCitations = [
          ...matches.map((m) => ({ id: m.id, metadata: m.metadata })),
          ...webCitations
      ];

      res.json({
        answer: googleRes.text || "I found some results but couldn't generate a summary.",
        citations: allCitations
      });

    } else {
      const completionRes = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${config.openaiKey}` },
        body: JSON.stringify({
          model: "gpt-5-nano",
          messages: [
            { role: "system", content: `You are Nebula Assistant. Cite sources as [Source Name].\n\nContext:\n${context}` },
            ...messages.map((m) => ({ role: m.role === 'model' ? 'assistant' : 'user', content: m.content }))
          ]
        })
      });
      const completionJson = await completionRes.json();
      res.json({
        answer: completionJson.choices?.[0]?.message?.content || "Sorry, I could not generate an answer.",
        citations: matches.map((m) => ({ id: m.id, metadata: m.metadata }))
      });
    }

  } catch (e) {
    console.error('QUERY_API_ERROR: An unexpected error occurred in the endpoint.');
    console.error(e); // Log the full error
    res.status(500).json({ error: e.message, trace: e.stack });
  }
});

// HARD DELETE: Remove all vectors for a project
app.post('/api/delete_project_vectors', async (req, res) => {
    try {
        const { username, projectId } = req.body;
        if (!username || !projectId) {
            return res.status(400).json({ error: "Username and Project ID required" });
        }
        const host = await getPineconeHost();
        const namespaces = [`u_${username}_p_${projectId}`, `summary_u_${username}_p_${projectId}`];
        await Promise.all(namespaces.map(ns => 
            fetch(`https://${host}/vectors/delete`, {
                method: "POST",
                headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey || '' },
                body: JSON.stringify({ deleteAll: true, namespace: ns })
            })
        ));
        res.json({ success: true, message: `Vectors deleted for project ${projectId}` });
    } catch (e) {
        console.error("Delete Error:", e);
        res.status(200).json({ success: false, error: e.message }); 
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`[SYS] Server Active on ${PORT}`));
