import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { GoogleGenAI } from "@google/genai";
// In Node 18+ native fetch is available, but we import it just in case of environment differences
import fetch from 'node-fetch';

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increase limit for large payloads

// --- Configuration ---
const config = {
    pineconeKey: process.env.PINECONE_API_KEY,
    pineconeHost: process.env.PINECONE_HOST,
    tavilyKey: process.env.TAVILY_API_KEY,
    openaiKey: process.env.OPENAI_API_KEY,
    geminiKey: process.env.GEMINI_API_KEY
};

// --- Helpers ---

// Initialize Google GenAI Client
const getAI = (req) => {
    // Check for client-provided key in headers, fallback to server env
    const clientKey = req.headers['x-gemini-api-key'];
    const key = clientKey || config.geminiKey;
    
    if (!key) {
        throw new Error("Gemini API Key is missing. Please check your configuration.");
    }
    
    return new GoogleGenAI({ apiKey: key });
};

// Get Pinecone Host URL
const getPineconeHost = async () => {
    if (!config.pineconeHost) return "your-index-host.pinecone.io"; // Fallback or throw
    return config.pineconeHost;
};

// Parse JSON from LLM response (handling markdown code blocks)
const parseLlmJson = (text) => {
    try {
        let clean = text.replace(/```json/g, '').replace(/```/g, '').trim();
        return JSON.parse(clean);
    } catch (e) {
        console.warn("LLM JSON Parse Error:", e);
        return {};
    }
};

// Generate Embedding using OpenAI
const getOpenAIEmbedding = async (text) => {
    if (!config.openaiKey) throw new Error("OpenAI API Key not configured for embeddings.");
    
    const response = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${config.openaiKey}`
        },
        body: JSON.stringify({
            model: "text-embedding-3-small",
            input: text
        })
    });

    if (!response.ok) {
        const err = await response.text();
        throw new Error(`OpenAI Embedding Error: ${err}`);
    }

    const data = await response.json();
    return data.data[0].embedding;
};

// --- API Routes ---

// 1. EXTRACT: Handle File Uploads (Text & Images via Gemini)
app.post('/api/extract', upload.array('files'), async (req, res) => {
    console.log('[API] /api/extract called');
    try {
        const ai = getAI(req);
        const files = req.files;
        const processedDocs = [];

        for (const file of files) {
            const base64Data = file.buffer.toString('base64');
            const mimeType = file.mimetype;
            
            // For PDFs or Images, we use Gemini Vision to extract text
            if (mimeType.includes('pdf') || mimeType.includes('image')) {
                const response = await ai.models.generateContent({
                    model: 'gemini-2.5-flash',
                    contents: {
                        parts: [
                            { inlineData: { mimeType: mimeType, data: base64Data } },
                            { text: "Extract all text from this document verbatim. If it is a spreadsheet or table, format it as CSV." }
                        ]
                    }
                });
                processedDocs.push({
                    id: crypto.randomUUID(),
                    name: file.originalname,
                    type: 'text/plain',
                    content: response.text || "",
                    size: file.size,
                    chunks: []
                });
            } else {
                // Plain text processing
                processedDocs.push({
                    id: crypto.randomUUID(),
                    name: file.originalname,
                    type: mimeType,
                    content: file.buffer.toString('utf-8'),
                    size: file.size,
                    chunks: []
                });
            }
        }
        res.json({ docs: processedDocs });
    } catch (e) {
        console.error(e);
        res.status(500).json({ error: e.message });
    }
});

// 2. INDEX: Generate Embeddings & Upsert to Pinecone
app.post('/api/index', async (req, res) => {
    console.log('[API] /api/index called');
    try {
        const { docs, username, projectId } = req.body;
        const host = await getPineconeHost();
        const namespace = `u_${username}_p_${projectId}`;
        
        const vectors = [];
        
        // Simple chunking strategy
        for (const doc of docs) {
            const chunks = doc.content.match(/[\s\S]{1,1000}/g) || [];
            for (let i = 0; i < chunks.length; i++) {
                const chunkText = chunks[i];
                const embedding = await getOpenAIEmbedding(chunkText);
                
                vectors.push({
                    id: `${doc.id}_chunk_${i}`,
                    values: embedding,
                    metadata: {
                        text: chunkText,
                        source: doc.name,
                        docId: doc.id
                    }
                });
            }
        }

        // Batch upsert to Pinecone (max 100 vectors per request usually recommended)
        const batchSize = 50;
        for (let i = 0; i < vectors.length; i += batchSize) {
            const batch = vectors.slice(i, i + batchSize);
            await fetch(`https://${host}/vectors/upsert`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Api-Key": config.pineconeKey
                },
                body: JSON.stringify({
                    vectors: batch,
                    namespace: namespace
                })
            });
        }

        res.json({ success: true, chunksProcessed: vectors.length });
    } catch (e) {
        console.error(e);
        res.status(500).json({ error: e.message });
    }
});

// 3. ANALYZE GRAPH: Generate Knowledge Graph
app.post('/api/analyze_graph', async (req, res) => {
    console.log('[API] /api/analyze_graph called');
    try {
        const ai = getAI(req);
        const { fullText, username, projectId } = req.body;

        const response = await ai.models.generateContent({
            model: 'gemini-3-pro-preview',
            contents: `Analyze the following text and extract a Knowledge Graph.
            Return a JSON object with "nodes" (id, label, group) and "links" (source, target, relation, value).
            Group 1 are main entities, Group 2 are secondary.
            Text: ${fullText.substring(0, 30000)}`, // Limit context
            config: { responseMimeType: "application/json" }
        });

        const graphData = parseLlmJson(response.text);
        
        // Optional: Generate Summary of the graph for the Summary Namespace
        // ... (Skipped for brevity, but logic would go here)

        res.json(graphData);
    } catch (e) {
        console.error(e);
        res.status(500).json({ error: e.message });
    }
});

// 4. QUERY: Dual-Source RAG
app.post('/api/query', async (req, res) => {
  console.log('[API] /api/query called.');
  try {
    const ai = getAI(req);
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
      
      results.forEach(result => {
          if (result.matches && Array.isArray(result.matches)) {
              matches.push(...result.matches);
          }
      });

      context = matches.map((m) =>
        `[Source: ${m.metadata.source || 'Community Summary'}]\n${m.metadata.text}`
      ).join("\n\n---\n\n");
    }

    // --- STAGE 2: WEB SEARCH ---
    let webContext = "";
    let webCitations = [];
    if (useWebSearch) {
        console.log(`[API] Performing live web search for: "${queryText}"`);
        try {
            if (!config.tavilyKey) throw new Error("Tavily API key is not configured.");
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
                        tavilyData.results.map(r => `[Source: ${r.url}]\n${r.content}`).join("\n\n") + "\n\n";
                    webCitations = tavilyData.results.map(r => ({ type: 'web', url: r.url, title: r.title }));
                }
            }
        } catch (e) {
            console.warn("Web search failed:", e.message);
        }
    }

    // --- STAGE 3: ANSWERING ---
    const fullContext = webContext + context;
    const systemPrompt = `You are Nebula Assistant. Answer EXCLUSIVELY based on the CONTEXT below.
If the answer is not in the context, say "Ich kann diese Frage basierend auf den vorliegenden Dokumenten nicht beantworten".
Answer in German. Cite sources using [Source Name] or [URL].

CONTEXT:
${fullContext || "No context found."}`;

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

    const completionJson = await completionRes.json();
    const finalAnswer = completionJson.choices?.[0]?.message?.content || "Sorry, I could not generate an answer.";
    
    const allCitations = [
        ...matches.map(m => ({ type: 'doc', id: m.id, metadata: m.metadata })),
        ...webCitations
    ];

    res.json({
      answer: finalAnswer,
      citations: allCitations,
      diagnostics: { webSearch: { used: useWebSearch, results: webCitations } }
    });

  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

// 5. DELETE: Cleanup Vectors
app.post('/api/delete_project_vectors', async (req, res) => {
    try {
        const { username, projectId } = req.body;
        const host = await getPineconeHost();
        const namespace = `u_${username}_p_${projectId}`;
        
        await fetch(`https://${host}/vectors/delete`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Api-Key": config.pineconeKey
            },
            body: JSON.stringify({ deleteAll: true, namespace })
        });
        
        res.json({ success: true });
    } catch (e) {
        console.error(e);
        res.status(500).json({ error: e.message });
    }
});

// 6. AUTH CONFIG: Serve Public IDs
app.get('/api/auth-config', (req, res) => {
    // DO NOT expose secrets here, only Public IDs for client-side OAuth
    res.json({
        clientId: process.env.GOOGLE_CLIENT_ID,
        appId: process.env.GOOGLE_APP_ID,
        apiKey: process.env.GOOGLE_PICKER_KEY // Public API Key for Picker
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
