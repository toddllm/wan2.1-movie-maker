#!/usr/bin/env python3
import os
import json
import glob
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Base directory for voice samples
VOICE_BASE_DIR = os.path.expanduser("~/movie_maker/hdmy5movie_voices")

class FileStatusHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Parse URL
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Handle API requests
        if path == "/api/voice-status":
            self.send_voice_status()
        elif path == "/api/sample-stats":
            self.send_sample_stats()
        else:
            # Regular file serving
            return SimpleHTTPRequestHandler.do_GET(self)
    
    def send_sample_stats(self):
        """Send overall stats about sample generation"""
        try:
            # Check explore directory
            explore_dir = os.path.join(VOICE_BASE_DIR, "explore")
            samples_pattern = os.path.join(explore_dir, "speaker_*_temp_*_topk_*.wav")
            
            # Get list of all generated sample files
            generated_files = glob.glob(samples_pattern)
            
            # Count unique speakers, temperatures, and topk values
            speakers = set()
            temperatures = set()
            topk_values = set()
            
            for file in generated_files:
                base = os.path.basename(file)
                parts = base.split('_')
                
                try:
                    speaker = int(parts[1])
                    temp = float(parts[3])
                    topk = int(parts[5].split('.')[0])
                    
                    speakers.add(speaker)
                    temperatures.add(temp)
                    topk_values.add(topk)
                except (IndexError, ValueError):
                    # Skip malformed filenames
                    continue
            
            # Create stats object
            stats = {
                "total_samples": len(generated_files),
                "total_expected": 115,  # Hardcoded based on progress info
                "progress_percent": round(len(generated_files) / 115 * 100, 1),
                "unique_speakers": sorted(list(speakers)),
                "unique_temperatures": sorted(list(temperatures)),
                "unique_topk": sorted(list(topk_values)),
                "last_modified": max([os.path.getmtime(f) for f in generated_files]) if generated_files else 0
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())
            
        except Exception as e:
            self.send_error(500, str(e))
    
    def send_voice_status(self):
        """Send the status of all expected voice samples"""
        try:
            # Define all expected combinations
            speakers = [0, 1, 2, 3, 4, 5, 6, 7]
            temperatures = [0.5, 0.75, 0.9, 1.1, 1.3]
            topk_values = [20, 50, 80]
            
            # Base directory for explore samples
            explore_dir = os.path.join(VOICE_BASE_DIR, "explore")
            
            # Prepare results
            results = []
            
            # Check each combination
            for speaker in speakers:
                for temp in temperatures:
                    for topk in topk_values:
                        filename = f"speaker_{speaker}_temp_{temp}_topk_{topk}.wav"
                        filepath = os.path.join(explore_dir, filename)
                        exists = os.path.exists(filepath)
                        
                        if exists:
                            size = os.path.getsize(filepath)
                            mtime = os.path.getmtime(filepath)
                        else:
                            size = 0
                            mtime = 0
                        
                        results.append({
                            "speaker": speaker,
                            "temperature": temp,
                            "topk": topk,
                            "filename": filename,
                            "exists": exists,
                            "size_kb": round(size / 1024, 1) if exists else 0,
                            "modified": mtime
                        })
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(results).encode())
            
        except Exception as e:
            self.send_error(500, str(e))

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, FileStatusHandler)
    print(f"Starting file status server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server() 