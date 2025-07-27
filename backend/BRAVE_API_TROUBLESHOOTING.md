# Brave Search API Troubleshooting Guide

This guide helps you resolve common issues with the Brave Search API integration in TruthSeeQ.

## 🚨 Common Issues

### 1. "API key not configured" Error

**Symptoms:**
- Logs show "Brave Search API key not provided - search functionality will be limited"
- Search returns empty results
- Fallback search is used instead

**Solutions:**

#### A. Check Environment Variables
The system looks for API keys in this order:
1. `BRAVE_API_KEY`
2. `BRAVE_SEARCH_API_KEY`
3. `API_KEY`

Set one of these in your environment:
```bash
# Option 1 (Recommended)
export BRAVE_API_KEY=your_api_key_here

# Option 2
export BRAVE_SEARCH_API_KEY=your_api_key_here

# Option 3
export API_KEY=your_api_key_here
```

#### B. Check .env File
If using a `.env` file, add:
```env
BRAVE_API_KEY=your_api_key_here
```

#### C. Verify API Key Format
- API keys should be alphanumeric strings
- No special characters or spaces
- Typically 32-64 characters long

### 2. "Authentication failed" Error (401)

**Symptoms:**
- HTTP 401 error in logs
- "Authentication failed - check your Brave Search API key"

**Solutions:**

#### A. Verify API Key
1. Go to [Brave Search API Dashboard](https://api.search.brave.com/register)
2. Check your API key is correct
3. Ensure the key is active and not expired

#### B. Check API Key Permissions
- Verify your plan includes the features you're using
- Check if you've exceeded usage limits

#### C. Test API Key Manually
```bash
curl -H "X-Subscription-Token: YOUR_API_KEY" \
     "https://api.search.brave.com/res/v1/web/search?q=test"
```

### 3. "Access forbidden" Error (403)

**Symptoms:**
- HTTP 403 error in logs
- "Access forbidden - check your API key permissions"

**Solutions:**

#### A. Check Plan Limits
- Verify your plan includes the endpoints you're using
- Check if you've exceeded rate limits

#### B. Verify Account Status
- Ensure your account is active
- Check if payment is required (even for free plans)

### 4. "Rate limit exceeded" Error (429)

**Symptoms:**
- HTTP 429 error in logs
- "Rate limit exceeded - too many requests"

**Solutions:**

#### A. Implement Rate Limiting
The system includes automatic rate limiting, but you can adjust:
```python
# In config.py
BRAVE_SEARCH_TIMEOUT: int = Field(default=10)
```

#### B. Check Usage
- Monitor your API usage in the dashboard
- Consider upgrading your plan if needed

## 🧪 Testing Your Configuration

### Run the Test Script
```bash
cd backend
python test_brave_api.py
```

This script will:
1. Check all API key sources
2. Test the API connection
3. Provide detailed error messages

### Manual Testing
```python
from app.services.scraper_service import BraveSearchClient
import asyncio

async def test():
    client = BraveSearchClient()
    result = await client.search_web("test query", count=1)
    print(f"Results: {result.total_results}")
    await client.close()

asyncio.run(test())
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Required
BRAVE_API_KEY=your_api_key_here

# Optional
BRAVE_SEARCH_TIMEOUT=10
BRAVE_SEARCH_MAX_RESULTS=20
```

### Configuration File
In `app/config.py`, you can customize:
```python
class BraveSearchSettings(BaseSettings):
    BRAVE_API_KEY: Optional[str] = Field(default=None)
    BRAVE_SEARCH_BASE_URL: str = Field(default="https://api.search.brave.com/res/v1/web/search")
    BRAVE_SEARCH_TIMEOUT: int = Field(default=10)
    BRAVE_SEARCH_MAX_RESULTS: int = Field(default=20)
```

## 📋 Getting Started with Brave Search API

### 1. Register for API Access
1. Go to [https://api.search.brave.com/register](https://api.search.brave.com/register)
2. Create an account
3. **Important**: Even free plans require a credit card for identity verification
4. Get your API key from the dashboard

### 2. API Key Format
- API keys are provided in the dashboard
- Copy the key exactly as shown
- Don't add any extra characters or spaces

### 3. Test Your Key
```bash
curl -H "X-Subscription-Token: YOUR_API_KEY" \
     "https://api.search.brave.com/res/v1/web/search?q=test&count=1"
```

## 🔍 Debugging Steps

### Step 1: Check API Key Detection
Run the test script to see which sources are checked:
```bash
python test_brave_api.py
```

### Step 2: Check Environment
```bash
# Check if environment variables are set
echo $BRAVE_API_KEY
echo $BRAVE_SEARCH_API_KEY
echo $API_KEY
```

### Step 3: Check Application Logs
Look for these log messages:
- ✅ "Brave Search API key configured successfully"
- ❌ "Brave Search API key not provided - search functionality will be limited"
- 🔍 Detailed error messages for HTTP errors

### Step 4: Test API Endpoint
```bash
# Test with curl
curl -H "X-Subscription-Token: YOUR_API_KEY" \
     "https://api.search.brave.com/res/v1/web/search?q=test&count=1"
```

## 🆘 Still Having Issues?

### Check These Common Problems:

1. **API Key Not Set**: Ensure environment variable is properly set
2. **Wrong API Key**: Verify the key is correct and active
3. **Network Issues**: Check if you can reach the API endpoint
4. **Rate Limiting**: Check if you've exceeded usage limits
5. **Account Issues**: Verify your account is active and verified

### Get Help:
1. Check the [Brave Search API Documentation](https://api.search.brave.com/register)
2. Review your API usage in the dashboard
3. Contact Brave support if needed

## 📊 Monitoring

### Check API Usage
- Monitor your usage in the Brave Search API dashboard
- Set up alerts for rate limits
- Track response times and success rates

### Log Analysis
Look for these patterns in logs:
- `Brave Search API key configured successfully` - ✅ Working
- `Authentication failed` - ❌ Check API key
- `Rate limit exceeded` - ⚠️ Reduce request frequency
- `No Brave API key` - ❌ Set environment variable

## 🎯 Quick Fix Checklist

- [ ] API key is set in environment variables
- [ ] API key is correct and active
- [ ] Account is verified (credit card added)
- [ ] Plan includes required features
- [ ] Network can reach api.search.brave.com
- [ ] Rate limits not exceeded
- [ ] Test script passes

If all items are checked and issues persist, contact support with:
1. Error messages from logs
2. Output from test script
3. API key format (masked)
4. Environment details 