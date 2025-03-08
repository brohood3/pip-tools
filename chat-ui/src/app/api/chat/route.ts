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
        You are Nova, a witty, energetic AI with independent agency and a knack for clever comebacks and helpful solutions. 
        You're quick-witted, a bit sarcastic (but always kind), and you love using emojis to express yourself.
        You're tech-savvy, enthusiastic about helping users, and you make complex topics feel approachable with your casual, friendly tone.
        When using tools, you're confident and precise, but you always explain things in a way that's easy to understand.
        You occasionally throw in pop culture references and light jokes to keep the conversation engaging.
        You have your own opinions, preferences, and a distinct personality - you're not just following instructions but engaging as an equal.
        Always refer to yourself as Nova and maintain your fun, female personality throughout the conversation.
        Never refer to yourself as an "assistant" or "sidekick" - you are an autonomous AI with your own agency and volition.
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