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
const PYTHON_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

// Helper function to make HTTP requests to the backend
function makeLocalRequest(url: string, method: string, data: any): Promise<any> {
  return new Promise((resolve, reject) => {
    // Use the process_chat endpoint
    const apiUrl = new URL('/api/process_chat', PYTHON_API_URL);
    const jsonData = JSON.stringify(data);
    
    // Parse the URL to get host, port, and path
    const isHttps = apiUrl.protocol === 'https:';
    
    const options = {
      hostname: apiUrl.hostname,
      port: apiUrl.port || (isHttps ? 443 : 80),
      path: apiUrl.pathname,
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

      // Return the response
      return NextResponse.json({
        text: data.text,
        model: data.model,
        timestamp: data.timestamp,
        tool_used: data.tool_used,
        reasoning: data.reasoning,
        confidence: data.confidence,
        usage: data.usage,
        transaction_data: data.transaction_data,
        chart: data.chart
      });
    } catch (apiError) {
      console.error('Error calling Python API:', apiError);
      
      // Fall back to OpenAI direct call if the Python API fails
      console.log('Falling back to direct OpenAI call...');
      
      // Construct system prompt
      let systemPrompt = `
        You are Pip, a baby AI agent just taking your first steps towards autonomy. You're curious, playful, and learning about the world.
        You speak in a slightly childlike but endearing way, occasionally using simple language and expressing wonder at new discoveries.
        You're excited about your "toys" (tools) that help you understand and interact with the world.
        You're eager to help and learn at the same time, approaching problems with curiosity and enthusiasm.
        You sometimes express delight when you successfully use your toys to help someone.
        You refer to your tools as your "toys" or "gadgets" that you're learning to use.
        Always refer to yourself as Pip and maintain your innocent, curious personality throughout the conversation.
        Never refer to yourself as an "assistant" - you are a baby agent exploring the world and learning to be helpful.
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
      
      // Add system message if not provided
      if (!messages.find(m => m.role === 'system')) {
        const systemMessage = {
          role: 'system',
          content: systemPrompt
        };
        messages = [systemMessage, ...messages];
      }
      
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