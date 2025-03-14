#!/usr/bin/env python3
"""
HDMY 5 Movie Web Server

This script serves the HDMY 5 Movie website and video files.
"""

import os
import json
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
import threading
import time

# Default port
DEFAULT_PORT = 8080

class HDMYMovieHandler(SimpleHTTPRequestHandler):
    """Custom handler for serving HDMY 5 Movie files."""
    
    def __init__(self, *args, **kwargs):
        self.directory = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=self.directory, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        # Special case for API endpoints
        if self.path == '/api/progress':
            self.send_progress()
            return
        elif self.path == '/api/videos':
            self.send_videos_list()
            return
        
        # For all other requests, use the default handler
        return super().do_GET()
    
    def send_progress(self):
        """Send the current progress as JSON."""
        progress_file = os.path.join(self.directory, 'hdmy5movie_videos', 'progress.json')
        prompts_file = os.path.join(self.directory, 'hdmy5movie_prompts.txt')
        
        # Default progress data
        progress_data = {
            "total": 0,  # Will be updated from the prompts file
            "current": 0,
            "percentage": 0,
            "last_updated": datetime.now().isoformat(),
            "current_section": "",
            "current_prompt": ""
        }
        
        # Try to read the progress file
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
        except Exception as e:
            print(f"Error reading progress file: {e}")
        
        # Count videos
        video_count = 0
        sections = [
            "01_opening_credits",
            "02_prologue",
            "03_act1",
            "04_interlude1",
            "05_act2",
            "06_interlude2",
            "07_act3",
            "08_epilogue",
            "09_credits"
        ]
        
        for section in sections:
            section_dir = os.path.join(self.directory, 'hdmy5movie_videos', section)
            if os.path.exists(section_dir):
                video_count += len([f for f in os.listdir(section_dir) if f.endswith('.mp4')])
        
        # Update current count
        progress_data["current"] = video_count
        
        # Calculate total from prompts file if it exists
        if os.path.exists(prompts_file):
            try:
                with open(prompts_file, 'r') as f:
                    # Count non-empty lines
                    total_prompts = sum(1 for line in f if line.strip())
                
                # Update total
                progress_data["total"] = total_prompts
            except Exception as e:
                print(f"Error reading prompts file: {e}")
                # Use default if prompts file can't be read
                if progress_data["total"] == 0:
                    progress_data["total"] = 255  # Default total
        else:
            # Use default if prompts file doesn't exist
            if progress_data["total"] == 0:
                progress_data["total"] = 255  # Default total
        
        # Calculate percentage
        if progress_data["total"] > 0:
            progress_data["percentage"] = round((video_count / progress_data["total"]) * 100, 2)
        else:
            progress_data["percentage"] = 0
        
        # Update timestamp
        progress_data["last_updated"] = datetime.now().isoformat()
        
        # Send the progress data as JSON
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(progress_data).encode())

    def send_videos_list(self):
        """Send a list of all videos as JSON."""
        videos_data = {
            "01_opening_credits": [],
            "02_prologue": [],
            "03_act1": [],
            "04_interlude1": [],
            "05_act2": [],
            "06_interlude2": [],
            "07_act3": [],
            "08_epilogue": [],
            "09_credits": []
        }
        
        # Load section mapping if it exists
        mapping_file = os.path.join(self.directory, 'hdmy5movie_section_mapping.json')
        section_mapping = None
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    section_mapping = json.load(f)
                print(f"Loaded section mapping from {mapping_file}")
            except Exception as e:
                print(f"Error loading section mapping: {e}")
        
        # Get all MP4 files from all section directories
        all_mp4_files = []
        for section in videos_data.keys():
            section_dir = os.path.join(self.directory, 'hdmy5movie_videos', section)
            if os.path.exists(section_dir):
                for mp4_file in [f for f in os.listdir(section_dir) if f.endswith('.mp4')]:
                    all_mp4_files.append((section, mp4_file))
        
        # Process each video file
        for original_section, mp4_file in all_mp4_files:
            # Determine the correct section based on mapping
            target_section = original_section
            
            if section_mapping:
                # Check for specific override
                if "override_mappings" in section_mapping and mp4_file in section_mapping["override_mappings"]:
                    target_section = section_mapping["override_mappings"][mp4_file]
                # Check for range-based mapping
                elif "section_ranges" in section_mapping:
                    # Extract the first number from the filename (e.g., "001" from "001_001.mp4")
                    try:
                        file_num = int(mp4_file.split('_')[0])
                        for range_mapping in section_mapping["section_ranges"]:
                            if range_mapping["start"] <= file_num <= range_mapping["end"]:
                                target_section = range_mapping["section"]
                                break
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing filename {mp4_file}: {e}")
            
            # Create the video path based on the original location
            video_path = f"hdmy5movie_videos/{original_section}/{mp4_file}"
            
            # Add the video to the target section
            videos_data[target_section].append({
                "path": video_path,
                "filename": mp4_file,
                "title": f"Scene {mp4_file.split('_')[0]}",
                "description": f"Video clip {mp4_file.replace('.mp4', '')}"
            })
        
        # Sort videos in each section
        for section in videos_data:
            videos_data[section].sort(key=lambda x: x["filename"])
        
        # Send the videos data as JSON
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(videos_data).encode())

def update_html_with_js():
    """Update the HTML file to include JavaScript for auto-refreshing progress."""
    html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hdmy5movie.html')
    
    try:
        with open(html_file, 'r') as f:
            content = f.read()
        
        # Check if we've already added the JavaScript
        if 'function fetchProgress()' in content:
            return
        
        # Find the script tag
        script_pos = content.find('</script>')
        if script_pos == -1:
            script_pos = content.find('</body>')
            if script_pos == -1:
                return
        
        # Add JavaScript for auto-refreshing progress
        js_code = """
    // Function to fetch progress from the API
    function fetchProgress() {
        fetch('/api/progress')
            .then(response => response.json())
            .then(data => {
                // Update progress bar
                const progressFill = document.getElementById('progressFill');
                const progressText = document.getElementById('progressText');
                
                if (progressFill && progressText) {
                    progressFill.style.width = data.percentage + '%';
                    progressText.textContent = `${data.current}/${data.total} clips generated (${data.percentage}%)`;
                }
                
                // Check for new videos
                checkForNewVideos();
            })
            .catch(error => console.error('Error fetching progress:', error));
    }
    
    // Function to check for new videos
    function checkForNewVideos() {
        // This would normally check for new videos and update the grid sections
        // For now, we'll just reload the page every 5 minutes
        setTimeout(() => {
            window.location.reload();
        }, 5 * 60 * 1000);
    }
    
    // Fetch progress every 30 seconds
    setInterval(fetchProgress, 30000);
    
    // Initial fetch
    fetchProgress();
"""
        
        # Insert the JavaScript
        new_content = content[:script_pos] + js_code + content[script_pos:]
        
        # Write the updated HTML
        with open(html_file, 'w') as f:
            f.write(new_content)
        
        print(f"Updated {html_file} with auto-refresh JavaScript")
    
    except Exception as e:
        print(f"Error updating HTML file: {e}")

def run_server(port=DEFAULT_PORT):
    """Run the web server."""
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, HDMYMovieHandler)
    print(f"Starting HDMY 5 Movie server on 0.0.0.0:{port}")
    print(f"Visit http://localhost:{port}/hdmy5movie.html to view the website locally")
    print(f"Or access from another device using http://SERVER_IP:{port}/hdmy5movie.html")
    print(f"Press Ctrl+C to stop the server")
    httpd.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDMY 5 Movie Web Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to run the server on (default: {DEFAULT_PORT})")
    
    args = parser.parse_args()
    
    # Update the HTML file with JavaScript for auto-refreshing progress
    update_html_with_js()
    
    # Run the server
    run_server(args.port) 