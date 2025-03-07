import os
import time
import secrets
import re
from flask import Blueprint, request, jsonify, session
from siwe import (
    SiweMessage, 
    ExpiredMessage, 
    InvalidSignature, 
    MalformedSession, 
    NonceMismatch as NonceExpiredException
)
from web3 import Web3

# Initialize blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# In-memory nonce storage (use a database in production)
# Format: {address: {'nonce': str, 'expiration': int}}
nonce_store = {}
NONCE_EXPIRY = 300  # 5 minutes in seconds

# Session configuration
SESSION_DURATION = 86400  # 24 hours in seconds

@auth_bp.route('/nonce', methods=['GET'])
def get_nonce():
    """Generate a new nonce for authentication."""
    # Generate a random nonce
    nonce = secrets.token_hex(32)
    
    # Get the wallet address from the request (if provided for session renewal)
    address = request.args.get('address', '').lower()
    
    # Store the nonce with an expiration time
    if address:
        nonce_store[address] = {
            'nonce': nonce,
            'expiration': int(time.time()) + NONCE_EXPIRY
        }
    
    return jsonify({'nonce': nonce})

@auth_bp.route('/verify', methods=['POST'])
def verify_signature():
    """Verify a signed SIWE message and establish a session."""
    try:
        # Get message and signature from request
        data = request.json
        if not data or 'message' not in data or 'signature' not in data:
            return jsonify({'error': 'Missing message or signature'}), 400
        
        message = data['message']
        signature = data['signature']
        
        print(f"Received message: {message}")
        print(f"Received signature: {signature}")
        
        # Extract the address from the message first
        address_match = re.search(r'\n(0x[a-fA-F0-9]{40})\n', message, re.IGNORECASE)
        if not address_match:
            return jsonify({'error': 'Could not extract address from message'}), 400
        
        extracted_address = address_match.group(1)
        lookup_address = extracted_address.lower()
        
        # Extract nonce from message for our verification
        nonce_match = re.search(r'Nonce: ([a-f0-9]+)', message)
        if not nonce_match:
            return jsonify({'error': 'Could not extract nonce from message'}), 400
        
        message_nonce = nonce_match.group(1)
        
        # Check if we have a nonce for this address
        if lookup_address not in nonce_store:
            return jsonify({'error': 'No nonce found for this address'}), 400
        
        # Verify the message contains our nonce
        stored_nonce = nonce_store[lookup_address]['nonce']
        if message_nonce != stored_nonce:
            return jsonify({'error': f'Invalid nonce. Expected {stored_nonce}, got {message_nonce}'}), 400
        
        # Check if nonce is expired
        if nonce_store[lookup_address]['expiration'] < int(time.time()):
            del nonce_store[lookup_address]
            return jsonify({'error': 'Nonce expired'}), 400
        
        # Using eth_account instead of siwe library for signature verification
        from eth_account.messages import encode_defunct
        from eth_account import Account
        
        try:
            # Create the same message for verification (don't modify the original)
            msg = encode_defunct(text=message)
            recovered_address = Account.recover_message(msg, signature=signature)
            
            # Compare recovered address with the address in the message
            if recovered_address.lower() != lookup_address:
                print(f"Address mismatch: Recovered {recovered_address.lower()}, Expected {lookup_address}")
                return jsonify({'error': 'Invalid signature: recovered address does not match'}), 400
                
            print(f"Signature verification successful! Recovered address: {recovered_address}")
        except Exception as e:
            print(f"Signature verification failed: {str(e)}")
            return jsonify({'error': f'Invalid signature: {str(e)}'}), 400
        
        # Clean up the used nonce
        del nonce_store[lookup_address]
        
        # Extract chain ID from the message
        chain_id_match = re.search(r'Chain ID: (\d+)', message)
        chain_id = int(chain_id_match.group(1)) if chain_id_match else 1
        
        # Set session data
        session['auth'] = {
            'address': lookup_address,
            'chain_id': chain_id,
            'expiry': int(time.time()) + SESSION_DURATION
        }
        
        return jsonify({
            'success': True,
            'address': lookup_address
        })
    
    except ExpiredMessage:
        return jsonify({'error': 'Message expired'}), 400
    except InvalidSignature:
        return jsonify({'error': 'Invalid signature'}), 400
    except NonceExpiredException:
        return jsonify({'error': 'Nonce has expired'}), 400
    except MalformedSession:
        return jsonify({'error': 'Malformed message'}), 400
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({'error': f'Verification error: {str(e)}'}), 500

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """End the user's session."""
    session.pop('auth', None)
    return jsonify({'success': True})

@auth_bp.route('/session', methods=['GET'])
def session_status():
    """Get the current authentication status."""
    if 'auth' not in session:
        return jsonify({'authenticated': False})
    
    auth_data = session['auth']
    
    # Check if session has expired
    if auth_data['expiry'] < int(time.time()):
        session.pop('auth', None)
        return jsonify({'authenticated': False, 'reason': 'expired'})
    
    return jsonify({
        'authenticated': True,
        'address': auth_data['address'],
        'chain_id': auth_data['chain_id']
    }) 