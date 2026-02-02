import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { GoogleGenAI, Type } from '@google/genai';
import { createClient } from '@supabase/supabase-js';
import fetch from 'node-fetch';
import crypto from 'crypto';
import { google } from 'googleapis';

const app = express();
const upload = multer();

app.use(cors());
app.use(express.json({ limit: '100mb' }));

// --- Configuration ---
// Updated to check ALL possible variable names for keys to prevent "Missing Key" errors
const config = {
  openaiKey: process.env.OPENAI_API_KEY,
  pineconeKey: process.env.PINECONE_API || process.env.PINECONE_API_KEY,
  // PRIORITIZE: API_KEY (System Standard) -> GOOGLE_GENAI_KEY (Your Config) -> GEMINI_API_KEY (Fallback)
  geminiKey: process.env.API_KEY || process.env.GOOGLE_GENAI_KEY || process.env.GEMINI_API_KEY, 
  tavilyKey: process.env.TAVILY_API_KEY,
  supabaseUrl: process.env.SUPABASE_URL,
  supabaseKey: process.env.SUPABASE_KEY,
  indexName: process.env.INDEX_NAME || 'clean-user',
  // Google Auth Config
  googleClientId: process.env.GOOGLE_CLIENT_ID,
  googlePickerApiKey: process.env.GOOGLE_PICKER_API_KEY,
  googleAppId: process.env.GOOGLE_APP_ID
};

// --- DYNAMIC AI CLIENT ---
const getAI = (req) => {
    // Check custom header first, then fall back to env var
    const apiKey = req.headers['x-gemini-api-key'] || config.geminiKey;
    
    if (!apiKey) {
        throw new Error("No Gemini API Key provided. Please configure API_KEY or GOOGLE_GENAI_KEY on server.");
    }
    
    return new GoogleGenAI({ apiKey });
};

/**
 * CORE HELPERS
 */
const getHash = (text) => crypto.createHash('sha256').update(text).digest('hex');

async function getPineconeHost() {
  // If host is manually configured, use it
  if (process.env.PINECONE_HOST) return process.env.PINECONE_HOST;

  // Otherwise fetch from Pinecone API
  const response = await fetch(`https://api.pinecone.io/indexes/${config.indexName}`, {
    headers: { "Api-Key": config.pineconeKey || '' }
  });
  
  if (!response.ok) {
      console.warn("Failed to fetch Pinecone Host via API. using fallback if available.");
      return "index-host-placeholder.pinecone.io";
  }

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
      dimensions: 1024 // IMPORTANT: MATCH PINECONE INDEX DIMENSION
    })
  });
  const json = await response.json();
  return json.data[0].embedding;
}

/**
 * Safely parses JSON from an LLM's output, handling markdown wrappers.
 */
function parseLlmJson(text) {
    if (!text) return {};
    try {
        const cleanedText = text.replace(/^```json\s*/, '').replace(/```$/, '').trim();
        return JSON.parse(cleanedText);
    } catch (e1) {
        try {
            const match = text.match(/\{[\s\S]*\}/);
            if (match && match[0]) {
                return JSON.parse(match[0]);
            }
        } catch (e2) {
            console.error("Failed to parse LLM JSON:", text);
        }
    }
    return {}; 
}

/**
 * ENDPOINTS
 */

// Auth Config Endpoint
app.get('/api/auth-config', (req, res) => {
    res.json({
        clientId: config.googleClientId,
        apiKey: config.googlePickerApiKey,
        appId: config.googleAppId
    });
});

// Extract
app.post('/api/extract', upload.array('files'), async (req, res) => {
  try {
    const ai = getAI(req); 
    const extractedDocs = [];
    
    for (const file of req.files) {
      let text = "";
      if (file.mimetype.startsWith('image/') || file.mimetype === 'application/pdf') {
        const result = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
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
    console.error("Extract Error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// Ingest from Drive
app.post('/api/ingest-drive', async (req, res) => {
    try {
        const { oauthToken, fileIds } = req.body;
        if (!oauthToken || !fileIds || !Array.isArray(fileIds)) throw new Error("Missing token or fileIds");

        const auth = new google.auth.OAuth2();
        auth.setCredentials({ access_token: oauthToken });
        const drive = google.drive({ version: 'v3', auth });

        // Recursive helper to flatten folders
        const getAllFiles = async (items) => {
            let results = [];
            for (const item of items) {
                let meta = item;
                if (typeof item === 'string') {
                    try {
                        const res = await drive.files.get({ fileId: item, fields: 'id, name, mimeType, size' });
                        meta = res.data;
                    } catch (e) {
                         console.error(`Skipping ID ${item}: ${e.message}`);
                         continue;
                    }
                }

                if (meta.mimeType === 'application/vnd.google-apps.folder') {
                    let pageToken;
                    do {
                        const listRes = await drive.files.list({
                            q: `'${meta.id}' in parents and trashed = false`,
                            fields: 'nextPageToken, files(id, name, mimeType, size)',
                            pageSize: 100,
                            pageToken
                        });
                        const children = listRes.data.files || [];
                        if (children.length > 0) {
                            results = results.concat(await getAllFiles(children));
                        }
                        pageToken = listRes.data.nextPageToken;
                    } while (pageToken);
                } else {
                    results.push(meta);
                }
            }
            return results;
        };

        const flatFiles = await getAllFiles(fileIds);
        
        const processFile = async (meta) => {
            try {
                const fileId = meta.id;
                const name = meta.name || 'Untitled';
                const mimeType = meta.mimeType || 'application/octet-stream';
                const originalSize = meta.size ? parseInt(meta.size) : 0;

                let content = "";
                let finalType = mimeType;

                if (mimeType === 'application/vnd.google-apps.document') {
                    const response = await drive.files.export({ fileId, mimeType: 'text/plain' }, { responseType: 'text' });
                    content = response.data;
                    finalType = 'text/plain';
                } else if (mimeType === 'application/vnd.google-apps.spreadsheet') {
                    const response = await drive.files.export({ fileId, mimeType: 'text/csv' }, { responseType: 'text' });
                    content = response.data;
                    finalType = 'text/csv';
                } else if (mimeType === 'application/vnd.google-apps.presentation') {
                    const response = await drive.files.export({ fileId, mimeType: 'text/plain' }, { responseType: 'text' });
                    content = response.data;
                    finalType = 'text/plain';
                } else {
                    const response = await drive.files.get({ fileId, alt: 'media' }, { responseType: 'arraybuffer' });
                    
                    if (mimeType === 'application/pdf') {
                        const ai = getAI(req);
                        const base64 = Buffer.from(response.data).toString('base64');
                        const result = await ai.models.generateContent({
                            model: 'gemini-2.5-flash',
                            contents: {
                                parts: [
                                { text: "Extract text accurately." },
                                { inlineData: { data: base64, mimeType: 'application/pdf' } }
                                ]
                            }
                        });
                        content = result.text || "";
                    } else {
                        content = Buffer.from(response.data).toString('utf-8');
                    }
                }

                return {
                    id: getHash(name + Date.now()).slice(0, 12),
                    name: name,
                    content: content,
                    type: finalType,
                    size: originalSize > 0 ? originalSize : content.length 
                };
            } catch (fileErr) {
                console.error(`Failed to process file ${meta.name}:`, fileErr.message);
                return null; 
            }
        };

        const CONCURRENCY = 5; 
        const docs = [];
        
        for (let i = 0; i < flatFiles.length; i += CONCURRENCY) {
            const chunk = flatFiles.slice(i, i + CONCURRENCY);
            const results = await Promise.all(chunk.map(meta => processFile(meta)));
            docs.push(...results.filter(d => d !== null));
        }

        res.json({ docs });
    } catch (e) {
        console.error("Drive Ingest Error:", e);
        res.status(500).json({ error: e.message });
    }
});

// Indexing
app.post('/api/index', async (req, res) => {
  try {
    const { docs, username, projectId } = req.body;
    
    if (!projectId) throw new Error("Project ID is required for indexing.");

    const host = await getPineconeHost();
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

// Graph Generation
app.post('/api/analyze_graph', async (req, res) => {
  console.log('[API] /api/analyze_graph called.');
  try {
    const ai = getAI(req); 
    const { fullText, username, projectId } = req.body;

    if (!fullText || !username || !projectId) {
        return res.status(400).json({ error: "Missing fullText, username, or projectId." });
    }

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: `Analyze this text and generate a hierarchical Knowledge Graph. 
Identify entities (nodes) and relations (links). 
Group entities into 'communities' and provide a 200-word summary for each cluster in German.
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

    const graph = parseLlmJson(response.text || '{}');
    const communityVectors = [];
    const processedSummaries = new Set();

    if (graph.nodes) {
        for (const node of graph.nodes) {
          if (node.summary) {
            const summaryHash = getHash(node.summary);
            if (processedSummaries.has(summaryHash)) continue;
            processedSummaries.add(summaryHash);

            const emb = await getOpenAIEmbedding(node.summary);
            communityVectors.push({
              id: `comm_${summaryHash.slice(0, 16)}`,
              values: emb,
              metadata: {
                text: node.summary,
                community_id: `group_${node.group || 'general'}`,
                type: 'community_summary'
              }
            });
          }
        }
    }

    if (communityVectors.length > 0) {
      const host = await getPineconeHost();
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
    console.error('ANALYZE_GRAPH_ERROR:', e);
    res.status(500).json({ error: e.message });
  }
});

// Query RAG
app.post('/api/query', async (req, res) => {
  console.log('[API] /api/query called.');
  try {
    const ai = getAI(req); 
    const { messages, username, projectId, ragMode, useWebSearch } = req.body;
    
    if (!projectId) throw new Error("Project ID is required for querying.");

    const lastUserMsg = messages[messages.length - 1].content;
    const host = await getPineconeHost();

    // Stage 1: Resolver
    const resolverRes = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: `Act as a RAG Controller. Decide if this query needs context or history only.
Rules: If query contains technical entities or acronyms (e.g. IKEv2), specific facts, or asks for explanations, context is REQUIRED.
Return JSON: { "use_context": boolean, "resolved_task": "de-contextualized query" }
History: ${JSON.stringify(messages.slice(-3))}
Current Query: ${lastUserMsg}`,
      config: { responseMimeType: "application/json" }
    });

    const plan = parseLlmJson(resolverRes.text || '');
    let context = "";
    let matches = [];
    const queryText = plan.resolved_task || lastUserMsg;

    if (plan.use_context || useWebSearch) {
      const queryEmbedding = await getOpenAIEmbedding(queryText);
      const isGraph = ragMode === 'complex';
      
      const chunkNamespace = `u_${username}_p_${projectId}`;
      const summaryNamespace = `summary_u_${username}_p_${projectId}`;
      
      const queries = [];
      
      // 1. Always query Standard Document Chunks
      queries.push(
        fetch(`https://${host}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey || '' },
            body: JSON.stringify({
                vector: queryEmbedding,
                topK: 6,
                includeMetadata: true,
                namespace: chunkNamespace
            })
        }).then(r => r.json())
      );
      
      // 2. Also query Community Summaries if Graph is active
      if (isGraph) {
          queries.push(
            fetch(`https://${host}/query`, {
                method: "POST",
                headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey || '' },
                body: JSON.stringify({
                    vector: queryEmbedding,
                    topK: 3,
                    includeMetadata: true,
                    namespace: summaryNamespace
                })
            }).then(r => r.json())
          );
      }

      const results = await Promise.all(queries);
      
      results.forEach(data => {
          if (data.matches && Array.isArray(data.matches)) {
              matches.push(...data.matches);
          }
      });

      context = "--- USER DOCUMENTS CONTEXT ---\n" + matches.map((m) =>
        `[Source: ${m.metadata.source || 'Community Summary'}]\n${m.metadata.text}`
      ).join("\n\n---\n\n");
    }

    // Stage 2: Web Search
    let webContext = "";
    let webCitations = [];
    if (useWebSearch) {
        console.log(`[API] Performing live web search for: "${queryText}"`);
        try {
            if (!config.tavilyKey) throw new Error("Tavily API key is not configured on the server.");
            const tavilyResponse = await fetch("https://api.tavily.com/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    api_key: config.tavilyKey,
                    query: queryText,
                    max_results: 3,
                    include_raw_content: false,
                }),
            });
            if (tavilyResponse.ok) {
                const tavilyData = await tavilyResponse.json();
                if (tavilyData.results && tavilyData.results.length > 0) {
                    webContext = "--- WEB SEARCH RESULTS ---\n" + 
                        tavilyData.results.map(r => `[Source: ${r.url}]\n${r.content}`).join("\n\n") +
                        "\n--- END WEB SEARCH RESULTS ---\n\n";
                    webCitations = tavilyData.results.map(r => ({ type: 'web', url: r.url, title: r.title }));
                }
            }
        } catch (e) {
            console.warn("[API] Web search failed:", e.message);
        }
    }

    // Stage 3: Answer
    const fullContext = webContext + context;
    const systemPrompt = `You are Nebula Assistant.
- You are a strict RAG agent. You must answer the user's question **EXCLUSIVELY** based on the provided CONTEXT below.
- If the answer is not in the context, explicitly state (in German) that you cannot answer based on the provided data.
- Answer in German language.

CONTEXT:
${fullContext || "No context was found for this query."}`;

    const completionRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${config.openaiKey}` },
      body: JSON.stringify({
        model: "gpt-5-nano",
        messages: [
          { role: "system", content: systemPrompt },
          ...messages.map((m) => ({ role: m.role === 'model' ? 'assistant' : 'user', content: m.content }))
        ]
      })
    });

    if (!completionRes.ok) throw new Error(`Assistant API Failed: ${completionRes.statusText}`);
    
    const completionJson = await completionRes.json();
    const finalAnswer = completionJson.choices?.[0]?.message?.content || "Sorry, I could not generate an answer.";
    
    const allCitations = [...matches.map(m => ({ type: 'doc', id: m.id, metadata: m.metadata })), ...webCitations];

    res.json({
      answer: finalAnswer,
      citations: allCitations,
      diagnostics: { webSearch: { used: useWebSearch, results: webCitations } }
    });

  } catch (e) {
    console.error('QUERY_API_ERROR:', e);
    res.status(500).json({ error: e.message });
  }
});

// Delete Vectors
app.post('/api/delete_project_vectors', async (req, res) => {
    try {
        const { username, projectId } = req.body;
        const host = await getPineconeHost();
        const namespaces = [`u_${username}_p_${projectId}`, `summary_u_${username}_p_${projectId}`];
        await Promise.all(namespaces.map(ns => 
            fetch(`https://${host}/vectors/delete`, {
                method: "POST",
                headers: { "Content-Type": "application/json", "Api-Key": config.pineconeKey || '' },
                body: JSON.stringify({ deleteAll: true, namespace: ns })
            })
        ));
        res.json({ success: true });
    } catch (e) {
        console.error("Delete Error:", e);
        res.status(200).json({ success: false, error: e.message }); 
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`[SYS] Server Active on ${PORT}`);
    console.log('--- Configuration Check ---');
    console.log('GEMINI KEY DETECTED:', !!config.geminiKey);
    console.log('PINECONE KEY DETECTED:', !!config.pineconeKey);
    console.log('OPENAI KEY DETECTED:', !!config.openaiKey);
});
