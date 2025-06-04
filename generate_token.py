#!/usr/bin/env python3
"""
Standalone Kite Connect Authentication Helper
Place this in your project root directory
"""

import sys
import os

# Add the virtual environment's site-packages to Python path
venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'lib', 'python3.9', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)
else:
    print("Virtual environment not found. Please create it first:")
    print("python -m venv venv")
    print("source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("pip install -r requirements.txt")
    sys.exit(1)

try:
    from kiteconnect import KiteConnect
    import webbrowser
    import urllib.parse
    from getpass import getpass
    import json
    from datetime import datetime
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease install requirements:")
    print("pip install kiteconnect")
    sys.exit(1)

def generate_access_token():
    """Interactive script to generate Kite access token"""
    
    print("=== Kite Connect Authentication ===\n")
    
    # Get API credentials
    api_key = input("Enter your Kite API Key: ").strip()
    api_secret = getpass("Enter your Kite API Secret: ").strip()
    
    # Initialize Kite Connect
    kite = KiteConnect(api_key=api_key)
    
    # Get login URL
    login_url = kite.login_url()
    
    print(f"\n1. Opening login URL in your browser...")
    print(f"   If browser doesn't open, manually visit: {login_url}")
    
    # Open browser
    webbrowser.open(login_url)
    
    print("\n2. Login to Kite and authorize the app")
    print("3. After authorization, you'll be redirected to your redirect URL")
    print("4. Copy the entire redirect URL from your browser\n")
    
    # Get the redirect URL
    redirect_url = input("Paste the complete redirect URL here: ").strip()
    
    try:
        # Parse the request token from URL
        parsed_url = urllib.parse.urlparse(redirect_url)
        params = urllib.parse.parse_qs(parsed_url.query)
        
        if 'request_token' not in params:
            raise ValueError("request_token not found in URL")
        
        request_token = params['request_token'][0]
        print(f"\nRequest token extracted: {request_token[:10]}...")
        
        # Generate access token
        print("\nGenerating access token...")
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print("\n✅ Success! Access token generated")
        print(f"Access Token: {access_token}")
        
        # Save to file
        credentials = {
            "api_key": api_key,
            "api_secret": api_secret,
            "access_token": access_token,
            "generated_at": datetime.now().isoformat(),
            "user_id": data.get("user_id", ""),
            "user_name": data.get("user_name", "")
        }
        
        save_option = input("\nSave credentials to file? (y/n): ").strip().lower()
        if save_option == 'y':
            with open("kite_credentials.json", "w") as f:
                json.dump(credentials, f, indent=2)
            print("Credentials saved to kite_credentials.json")
            
            # Also create/update .env file
            env_content = f"""KITE_API_KEY={api_key}
KITE_API_SECRET={api_secret}
KITE_ACCESS_TOKEN={access_token}
LOG_LEVEL=INFO
"""
            with open(".env", "w") as f:
                f.write(env_content)
            print(".env file updated with new credentials")
        
        print("\n📝 Next steps:")
        print("1. Use these credentials in your .env file")
        print("2. Access tokens are valid for 1 day")
        print("3. You need to generate a new token daily")
        
        # Test the connection
        test_option = input("\nTest the connection? (y/n): ").strip().lower()
        if test_option == 'y':
            kite.set_access_token(access_token)
            profile = kite.profile()
            print(f"\n✅ Connection successful!")
            print(f"User: {profile['user_name']} ({profile['email']})")
            print(f"Broker: {profile['broker']}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nCommon issues:")
        print("1. Make sure you copied the complete URL including 'request_token' parameter")
        print("2. Request tokens are valid for only a few minutes")
        print("3. Each request token can be used only once")
        print("\nExample redirect URL format:")
        print("https://your-redirect-url.com/?request_token=xxxxx&action=login&status=success")

if __name__ == "__main__":
    generate_access_token()