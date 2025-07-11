# Deploy RudoWealth on Render.com

## Step-by-Step Deployment Guide

### 1. Sign Up for Render
- Go to [render.com](https://render.com)
- Sign up with your GitHub account
- Verify your email

### 2. Connect Your Repository
- Click "New +" → "Web Service"
- Connect your GitHub account if not already connected
- Select the `rudowealth` repository
- Choose the `main` branch

### 3. Configure the Service
Render will automatically detect the `render.yaml` configuration, but you can verify these settings:

**Basic Settings:**
- **Name**: `rudowealth`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements_render.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key (set this manually)
- All other variables are pre-configured in `render.yaml`

### 4. Set Environment Variables
In the Render dashboard, go to your service → Environment → Add Environment Variable:
- **Key**: `OPENAI_API_KEY`
- **Value**: Your actual OpenAI API key
- **Sync**: No (for security)

### 5. Deploy
- Click "Create Web Service"
- Render will automatically build and deploy your application
- The build process may take 5-10 minutes due to FAISS compilation

### 6. Access Your Application
- Once deployed, Render will provide a URL like: `https://rudowealth.onrender.com`
- Your application will be live and accessible

## Troubleshooting

### Build Issues
If the build fails:
1. Check the build logs in Render dashboard
2. Ensure all dependencies are in `requirements_render.txt`
3. Verify Python version compatibility

### Runtime Issues
If the app doesn't start:
1. Check the logs in Render dashboard
2. Verify environment variables are set correctly
3. Ensure the health check endpoint `/health` is working

### FAISS Issues
Render handles FAISS compilation well, but if you encounter issues:
1. The build may take longer than usual (5-15 minutes)
2. Check that you're using the free plan (sufficient for FAISS)
3. Monitor the build logs for any compilation errors

## Benefits of Render

✅ **FAISS Support**: Excellent support for complex Python packages
✅ **Free Tier**: Generous free tier with no image size limits
✅ **Auto-Deploy**: Automatic deployments on git push
✅ **Custom Domains**: Easy custom domain setup
✅ **SSL**: Automatic HTTPS certificates
✅ **Monitoring**: Built-in logging and monitoring

## Next Steps

After successful deployment:
1. Test all functionality (file upload, chat, etc.)
2. Set up a custom domain if desired
3. Monitor the application logs
4. Consider upgrading to paid plan for production use

Your RudoWealth application should now be live on Render with full FAISS functionality! 