# Environment Variables Setup

This document explains how to set up environment variables for the OATS Flask Chatbot application.

## Quick Setup

1. **Copy the environment template:**
   ```bash
   cp env_template.txt .env
   ```

2. **Edit the .env file** with your actual values:
   ```bash
   # Edit the .env file with your credentials
   nano .env  # or use your preferred editor
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

### Flask Configuration
- `FLASK_SECRET_KEY`: Secret key for Flask sessions (change in production)
- `FLASK_DEBUG`: Enable/disable debug mode (True/False)
- `FLASK_HOST`: Host to bind the Flask app to (default: 0.0.0.0)
- `FLASK_PORT`: Port to run the Flask app on (default: 5000)

### API Configuration
- `API_BASE_URL`: Base URL for the OATS backend API
- `API_LOGIN_URL`: Login endpoint path
- `API_LOGOUT_URL`: Logout endpoint path

### Authentication Credentials
- `API_EMAIL`: Email for API authentication
- `API_PASSWORD`: Password for API authentication

### External API Keys
- `GEMINI_API_KEY`: Google Gemini AI API key

### Session Configuration
- `SESSION_TYPE`: Session storage type (filesystem)
- `SESSION_PERMANENT`: Whether sessions are permanent (True/False)
- `PERMANENT_SESSION_LIFETIME_HOURS`: Session lifetime in hours

## Security Notes

⚠️ **IMPORTANT SECURITY CONSIDERATIONS:**

1. **Never commit .env files** - They are already in .gitignore
2. **Change default credentials** - Update API_EMAIL and API_PASSWORD
3. **Use strong secret keys** - Generate a strong FLASK_SECRET_KEY for production
4. **Secure API keys** - Keep your GEMINI_API_KEY secure
5. **Environment-specific configs** - Use different .env files for different environments

## Production Deployment

For production deployment:

1. Create a production-specific .env file
2. Set `FLASK_DEBUG=False`
3. Use a strong, randomly generated `FLASK_SECRET_KEY`
4. Ensure all API credentials are correct for production environment
5. Consider using environment-specific API endpoints

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError: No module named 'dotenv'**
   - Solution: `pip install python-dotenv`

2. **Environment variables not loading**
   - Ensure .env file is in the same directory as app.py
   - Check file permissions
   - Verify .env file format (no spaces around =)

3. **API authentication fails**
   - Verify API_EMAIL and API_PASSWORD are correct
   - Check API_BASE_URL is accessible
   - Ensure network connectivity to the API server

## Example .env File

```env
# Flask Application Configuration
FLASK_SECRET_KEY=your-super-secret-key-here
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# API Configuration
API_BASE_URL=https://your-api-server.com
API_LOGIN_URL=/rbca/token/
API_LOGOUT_URL=/login-api/logout/

# Authentication Credentials
API_EMAIL=your-email@company.com
API_PASSWORD=your-secure-password

# External API Keys
GEMINI_API_KEY=your-gemini-api-key

# Session Configuration
SESSION_TYPE=filesystem
SESSION_PERMANENT=False
PERMANENT_SESSION_LIFETIME_HOURS=2
```


