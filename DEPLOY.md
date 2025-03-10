# Deploying Pip AI Chat to Render

This guide walks you through deploying the Pip AI Chat application to [Render](https://render.com).

## Prerequisites

1. A Render account
2. OpenAI API key
3. Replicate API token (for video generation)
4. Your project code in a Git repository (GitHub, GitLab, etc.)

## Deployment Options

### Option 1: Using the Blueprint (Recommended)

1. Fork this repository to your GitHub account
2. Log in to your Render account
3. Click on "New" and select "Blueprint"
4. Connect your GitHub account and select your forked repository
5. Render will automatically detect the `render.yaml` file and set up the services
6. Add your environment variables (API keys) in the Render dashboard
7. Deploy!

### Option 2: Manual Setup

#### Backend Deployment

1. Log in to your Render account
2. Click on "New" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: pip-backend
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -k eventlet -w 1 wsgi:app --bind 0.0.0.0:$PORT`
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `REPLICATE_API_TOKEN`: Your Replicate API token
   - `SECRET_KEY`: A secure random string
   - `FLASK_ENV`: Set to `production`
   - `CORS_ORIGINS`: URL of your frontend (e.g., `https://pip-frontend.onrender.com`)
6. Click "Create Web Service"

#### Frontend Deployment

1. Click on "New" and select "Web Service"
2. Connect your GitHub repository
3. Configure the service:
   - **Name**: pip-frontend
   - **Runtime**: Node
   - **Build Command**: `cd chat-ui && npm install && npm run build`
   - **Start Command**: `cd chat-ui && npm start`
4. Add environment variables:
   - `NEXT_PUBLIC_API_URL`: URL of your backend (e.g., `https://pip-backend.onrender.com`)
   - `NEXT_PUBLIC_SOCKET_URL`: Same as above
5. Click "Create Web Service"

## Post-Deployment

1. Wait for both services to deploy successfully
2. Test the application by navigating to your frontend URL
3. If you encounter any issues, check the logs in the Render dashboard

## Scaling (Optional)

If you need to handle more traffic:

1. Go to your service in the Render dashboard
2. Click on "Settings"
3. Under "Instance Type", select a higher tier
4. For the backend, you can also increase the number of instances

## Monitoring

Render provides basic monitoring out of the box:

1. Go to your service in the Render dashboard
2. Click on "Metrics" to view CPU, memory, and network usage
3. Click on "Logs" to view application logs

## Troubleshooting

- **CORS Issues**: Ensure `CORS_ORIGINS` is set correctly in your backend environment variables
- **Socket Connection Issues**: Check that `NEXT_PUBLIC_SOCKET_URL` is correct and WebSocket connections are allowed
- **Build Failures**: Check the build logs for any dependency issues
- **Runtime Errors**: Check the application logs for error messages 