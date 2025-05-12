"""
CORS middleware for the Flask application
Enables cross-origin requests during development
"""

def add_cors_headers(response):
    """Add CORS headers to the response"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

def setup_cors(app):
    """Set up CORS for a Flask application"""
    app.after_request(add_cors_headers)
