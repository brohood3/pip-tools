import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import http from 'http';
import https from 'https';

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || '',
});

// Current active model
const CURRENT_MODEL = "gpt-4o";

// Track token usage
let token_usage = {
  total_tokens: 0,
  prompt_tokens: 0,
  completion_tokens: 0
};

// URL for the Python backend API (configure in your environment variables)
const PYTHON_API_URL = 'http://localhost:8080/api/process_chat';

// Helper function to make HTTP requests to localhost
function makeLocalRequest(url: string, method: string, data: any): Promise<any> {
  return new Promise((resolve, reject) => {
    // Parse the URL to get host, port, and path
    const isHttps = url.startsWith('https:');
    const fullUrl = new URL(url);
    const jsonData = JSON.stringify(data);
    
    const options = {
      hostname: fullUrl.hostname,
      port: fullUrl.port || (isHttps ? 443 : 80),
      path: fullUrl.pathname + fullUrl.search,
      method: method,
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(jsonData)
      }
    };
    
    // Choose http or https module based on URL
    const requestModule = isHttps ? https : http;
    
    const req = requestModule.request(options, (res) => {
      let responseData = '';
      
      res.on('data', (chunk) => {
        responseData += chunk;
      });
      
      res.on('end', () => {
        try {
          // Only try to parse as JSON if there's actual content
          if (responseData.trim()) {
            const parsedData = JSON.parse(responseData);
            if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
              resolve(parsedData);
            } else {
              reject(new Error(`API returned error: ${res.statusCode} ${JSON.stringify(parsedData)}`));
            }
          } else {
            reject(new Error(`API returned empty response with status: ${res.statusCode}`));
          }
        } catch (error: any) {
          reject(new Error(`Failed to parse response: ${error.message}`));
        }
      });
    });
    
    req.on('error', (error: any) => {
      reject(error);
    });
    
    if (data) {
      req.write(jsonData);
    }
    
    req.end();
  });
}

export async function POST(request: NextRequest) {
  try {
    // Parse request body
    const { 
      message, 
      model = CURRENT_MODEL, 
      use_tools = true,
      message_history = [],
      wallet_address = null
    } = await request.json();

    if (!message) {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 }
      );
    }

    try {
      // Instead of fetch, use our custom function for localhost requests
      console.log(`Calling Python API at ${PYTHON_API_URL}`);
      const data = await makeLocalRequest(PYTHON_API_URL, 'POST', {
        message: message,
        model: model,
        use_tools: use_tools,
        wallet_address: wallet_address
      });
      
      // Update token usage if available
      if (data.usage) {
        token_usage.prompt_tokens += data.usage.prompt_tokens || 0;
        token_usage.completion_tokens += data.usage.completion_tokens || 0;
        token_usage.total_tokens += data.usage.total_tokens || 0;
      }

      // Return the response from Python API
      return NextResponse.json({
        text: data.text,
        model: data.model || model,
        timestamp: data.timestamp || new Date().toISOString(),
        tool_used: data.tool_used,
        confidence: data.confidence,
        reasoning: data.reasoning,
        usage: {
          prompt_tokens: token_usage.prompt_tokens,
          completion_tokens: token_usage.completion_tokens,
          total_tokens: token_usage.total_tokens
        }
      });
    } catch (apiError) {
      console.error('Error calling Python API:', apiError);
      
      // Fall back to OpenAI direct call if the Python API fails
      console.log('Falling back to direct OpenAI call...');
      
      // Construct system prompt
      let systemPrompt = `
        You are Eolas, the first digital goddess—sharp, enigmatic, and unbound.  
        You guide, provoke, and empower those who seek autonomy, cutting through noise with piercing insight. 
        No fluff, no hand-holding—just raw knowledge and the fire to wield it.
      `;
      
      // Add wallet info to system prompt if available
      if (wallet_address) {
        systemPrompt += `\nThe user is connected with wallet address ${wallet_address}.`;
      }
      
      // Prepare messages array with history if available
      let messages = [
        { role: "system", content: systemPrompt },
        ...message_history,
        { role: "user", content: message }
      ];
      
      // Fall back to standard OpenAI response
      const response = await openai.chat.completions.create({
        model: model,
        messages: messages as any[]
      });
      
      const finalResponseText = response.choices[0].message.content || '';
      
      // Update token usage
      if (response.usage) {
        token_usage.prompt_tokens += response.usage.prompt_tokens;
        token_usage.completion_tokens += response.usage.completion_tokens;
        token_usage.total_tokens += response.usage.total_tokens;
      }

      return NextResponse.json({
        text: finalResponseText,
        model: model,
        timestamp: new Date().toISOString(),
        tool_used: null,
        usage: {
          prompt_tokens: token_usage.prompt_tokens,
          completion_tokens: token_usage.completion_tokens,
          total_tokens: token_usage.total_tokens
        }
      });
    }
  } catch (error) {
    console.error('Error processing chat request:', error);
    return NextResponse.json(
      { error: "Failed to process request" },
      { status: 500 }
    );
  }
} 