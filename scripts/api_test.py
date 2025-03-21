#!/usr/bin/env python3
"""
API connection test script for troubleshooting frontend-backend communication.
"""
import requests
import json
import sys
import argparse
from urllib.parse import urljoin

def test_api_endpoint(base_url, endpoint, method='GET', data=None, verbose=False):
    """Test an API endpoint and return status."""
    url = urljoin(base_url, endpoint)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    print(f"Testing {method} request to {url}")
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, timeout=10)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=data, timeout=10)
        else:
            print(f"Method {method} not supported")
            return False
        
        print(f"Status Code: {response.status_code}")
        
        if verbose:
            print("Response Headers:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")
        
        if 200 <= response.status_code < 300:
            try:
                response_data = response.json()
                print("\nResponse Data:")
                print(json.dumps(response_data, indent=2))
                return True
            except json.JSONDecodeError:
                print("\nResponse is not JSON. First 200 chars:")
                print(response.text[:200])
                return False
        else:
            print("\nError Response:")
            print(response.text[:200])
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test API endpoints')
    parser.add_argument('--base-url', '-b', default='https://ciwre-bae.campusad.msu.edu', 
                        help='Base URL for API (default: https://ciwre-bae.campusad.msu.edu)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    
    args = parser.parse_args()
    
    # First test the diagnostic API endpoint
    print("\n=== Testing Diagnostic Status Endpoint ===")
    status_ok = test_api_endpoint(args.base_url, '/api/diagnostic/status', verbose=args.verbose)
    
    if not status_ok:
        print("\n❌ API status check failed. Check network connectivity and server status.")
        sys.exit(1)
    
    # Test the echo endpoint
    print("\n=== Testing Echo Endpoint ===")
    echo_data = {'test': 'data', 'timestamp': 'now'}
    echo_ok = test_api_endpoint(args.base_url, '/api/diagnostic/echo', 
                                method='POST', data=echo_data, verbose=args.verbose)
    
    if not echo_ok:
        print("\n❌ Echo test failed. There might be issues with POST requests.")
    else:
        print("\n✅ Echo test successful!")
    
    # Test the model-settings simulation endpoint
    print("\n=== Testing Model Settings Simulation ===")
    model_data = {'site_no': '12345678', 'ls_resolution': 250, 'dem_resolution': 30}
    model_ok = test_api_endpoint(args.base_url, '/api/diagnostic/test-model-settings', 
                                method='POST', data=model_data, verbose=args.verbose)
    
    if not model_ok:
        print("\n❌ Model settings simulation failed. This suggests the actual model-settings endpoint may have issues.")
    else:
        print("\n✅ Model settings simulation successful!")
    
    # Print overall summary
    print("\n=== Test Summary ===")
    if status_ok and echo_ok and model_ok:
        print("✅ All tests passed! API communication appears to be working correctly.")
    else:
        print("❌ Some tests failed. Please check the logs for more information.")

if __name__ == "__main__":
    main()
