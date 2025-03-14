#!/usr/bin/env python3
"""
Feedback Server for CSM Voice Explorer

This script extends Python's HTTP server to handle feedback submissions
and store them in a JSON database file.
"""

import os
import json
import time
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs
import threading
import logging

# Configuration
PORT = 8000
FEEDBACK_FILE = os.path.expanduser("~/movie_maker/voice_feedback_db.json")
FEEDBACK_LOCK = threading.Lock()  # Lock for thread-safe file access
UPDATE_SCRIPT = os.path.expanduser("~/movie_maker/update_descriptions.py")
LOG_FILE = os.path.expanduser("~/movie_maker/feedback_server.log")

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FeedbackHandler(SimpleHTTPRequestHandler):
    """Handler for serving files and processing feedback submissions."""
    
    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        SimpleHTTPRequestHandler.end_headers(self)
    
    def do_POST(self):
        """Handle POST requests for feedback submission."""
        if self.path == '/save_feedback':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                feedback_data = json.loads(post_data.decode('utf-8'))
                self.save_feedback(feedback_data)
                
                # Run the update script in a separate thread
                threading.Thread(target=self.run_update_script).start()
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode('utf-8'))
            except Exception as e:
                # Send error response
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
                logging.error(f"Error processing feedback: {str(e)}")
        else:
            self.send_error(404, "Path not found")
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.end_headers()
    
    def save_feedback(self, feedback_data):
        """Save feedback data to the JSON database file."""
        with FEEDBACK_LOCK:
            # Load existing data
            all_feedback = self.load_feedback_db()
            
            # Add unique ID and timestamp if not present
            if 'id' not in feedback_data:
                feedback_data['id'] = f"feedback_{int(time.time())}_{len(all_feedback['feedback'])}"
            if 'timestamp' not in feedback_data:
                feedback_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add to database
            all_feedback['feedback'].append(feedback_data)
            all_feedback['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            all_feedback['count'] = len(all_feedback['feedback'])
            
            # Save back to file
            with open(FEEDBACK_FILE, 'w') as f:
                json.dump(all_feedback, f, indent=2)
            
            logging.info(f"Saved feedback: {feedback_data['sampleId']}")
            print(f"Saved feedback: {feedback_data['sampleId']}")
    
    def run_update_script(self):
        """Run the description update script."""
        try:
            if os.path.exists(UPDATE_SCRIPT):
                logging.info("Running description update script")
                result = subprocess.run([UPDATE_SCRIPT], capture_output=True, text=True)
                if result.returncode == 0:
                    logging.info(f"Update script completed successfully: {result.stdout.strip()}")
                else:
                    logging.error(f"Update script failed: {result.stderr.strip()}")
            else:
                logging.error(f"Update script not found: {UPDATE_SCRIPT}")
        except Exception as e:
            logging.error(f"Error running update script: {str(e)}")
    
    def load_feedback_db(self):
        """Load the feedback database or create it if it doesn't exist."""
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.error(f"Error reading {FEEDBACK_FILE}, creating new database")
                print(f"Error reading {FEEDBACK_FILE}, creating new database")
        
        # Create new database structure
        return {
            'feedback': [],
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'count': 0
        }
    
    def do_GET(self):
        """Handle GET requests, including a special endpoint for feedback data."""
        if self.path == '/get_feedback_summary':
            # Return the feedback summary
            with FEEDBACK_LOCK:
                all_feedback = self.load_feedback_db()
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(all_feedback).encode('utf-8'))
        else:
            # Handle normal file serving
            return SimpleHTTPRequestHandler.do_GET(self)

def run_server():
    """Run the HTTP server."""
    server_address = ('0.0.0.0', PORT)
    httpd = HTTPServer(server_address, FeedbackHandler)
    logging.info(f"Starting feedback server on 0.0.0.0:{PORT}...")
    logging.info(f"Feedback will be stored in {FEEDBACK_FILE}")
    print(f"Starting feedback server on 0.0.0.0:{PORT}...")
    print(f"Feedback will be stored in {FEEDBACK_FILE}")
    print("Press Ctrl+C to stop the server")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server() 