import time
from functools import wraps
from flask import session, jsonify, request, redirect, url_for

def auth_required(f):
    """
    Decorator for routes that require authentication.
    If the user is not authenticated, redirects to the login page or returns a 401 response.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is authenticated
        if 'auth' not in session:
            if request.content_type == 'application/json':
                return jsonify({'error': 'Authentication required'}), 401
            # For non-JSON requests, redirect to login page
            return redirect(url_for('login'))
        
        # Check if session has expired
        auth_data = session['auth']
        if auth_data['expiry'] < int(time.time()):
            session.pop('auth', None)
            if request.content_type == 'application/json':
                return jsonify({'error': 'Session expired'}), 401
            # For non-JSON requests, redirect to login page
            return redirect(url_for('login'))
        
        # User is authenticated
        return f(*args, **kwargs)
    
    return decorated_function 