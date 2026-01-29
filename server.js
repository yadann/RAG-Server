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
    const queryText = plan.resolved_task || lastUserMsg;

    if (plan.use_context) {
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
                topK: 6, // Fetch 6 chunks
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
                    topK: 3, // Fetch 3 summaries
                    includeMetadata: true,
                    namespace: summaryNamespace
                })
            }).then(r => r.json())
        );
      }

      const results = await Promise.all(queries);
      
      // Combine matches
      results.forEach(result => {
          if (result.matches && Array.isArray(result.matches)) {
              matches.push(...result.matches);
          }
      });

      context = "--- USER DOCUMENTS CONTEXT ---\n" + matches.map((m) =>
        `[Source: ${m.metadata.source || 'Community Summary'}]\n${m.metadata.text}`
      ).join("\n\n---\n\n");
    }

    // --- STAGE 2: WEB SEARCH (IF ENABLED) ---
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
            if (!tavilyResponse.ok) throw new Error(`Tavily API responded with status ${tavilyResponse.status}`);
            
            const tavilyData = await tavilyResponse.json();
            
            if (tavilyData.results && tavilyData.results.length > 0) {
                webContext = "--- REAL-TIME WEB SEARCH RESULTS ---\n" + 
                    tavilyData.results.map(r => `[Source: ${r.url}]\n${r.content}`).join("\n\n") +
                    "\n--- END WEB SEARCH RESULTS ---\n\n";
                
                webCitations = tavilyData.results.map(r => ({
                    type: 'web',
                    url: r.url,
                    title: r.title,
                }));
                 console.log(`[API] Found ${webCitations.length} web results from sites like ${tavilyData.results[0].url}`);
            }
        } catch (e) {
            console.warn("[API] Tavily web search failed:", e.message);
            webContext = `--- WEB SEARCH FAILED: ${e.message} ---\n`;
        }
    }

    // --- STAGE 3: UNIFIED ANSWERING ---
    const fullContext = webContext + context;
    const systemPrompt = `You are Nebula Assistant.
- You are a strict RAG agent. You must answer the user's question **EXCLUSIVELY** based on the provided CONTEXT below.
- **DO NOT** use your internal knowledge to answer questions not present in the context.
- If the answer is not in the context, explicitly state (in German) that you cannot answer based on the provided data ("Ich kann diese Frage basierend auf den vorliegenden Dokumenten nicht beantworten").
- If web search results are present, use them as valid context.
- Answer in German language.
- Cite your sources. For documents, use [Source Name]. For web pages, use the full URL as [https://...].

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

    if (!completionRes.ok) {
        const errorText = await completionRes.text();
        console.error('Error from gpt-5-nano API:', errorText);
        throw new Error(`Failed to get response from assistant AI: ${completionRes.statusText}`);
    }
    
    const completionJson = await completionRes.json();
    const finalAnswer = completionJson.choices?.[0]?.message?.content || "Sorry, I could not generate an answer.";
    
    const allCitations = [
        ...matches.map(m => ({ type: 'doc', id: m.id, metadata: m.metadata })),
        ...webCitations
    ];

    res.json({
      answer: finalAnswer,
      citations: allCitations,
      diagnostics: {
        webSearch: {
          used: useWebSearch,
          query: queryText,
          results: webCitations
        }
      }
    });

  } catch (e) {
    console.error('QUERY_API_ERROR: An unexpected error occurred in the endpoint.');
    console.error(e);
    res.status(500).json({ error: e.message, trace: e.stack });
  }
});
