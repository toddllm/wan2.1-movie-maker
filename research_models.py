#!/usr/bin/env python3
"""
Model Research Script for Movie Making

This script searches for and catalogs AI models that could be useful for movie making,
including models for video generation, image processing, audio processing, and text-to-speech.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests
import pandas as pd
from tqdm import tqdm

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_research.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_research")

class ModelResearcher:
    def __init__(self):
        self.categories = {
            "video_generation": [
                "text-to-video",
                "video-to-video",
                "image-to-video",
                "video-generation"
            ],
            "image_generation": [
                "text-to-image",
                "image-to-image",
                "stable-diffusion",
                "controlnet"
            ],
            "audio_processing": [
                "text-to-speech",
                "speech-to-text",
                "audio-to-audio",
                "music-generation",
                "voice-conversion"
            ],
            "multimodal": [
                "visual-question-answering",
                "image-text-generation",
                "video-text-generation",
                "multimodal"
            ],
            "language_models": [
                "text-generation",
                "text2text-generation",
                "summarization",
                "translation"
            ]
        }
        
        self.results_dir = "model_research"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def search_huggingface(self, category: str, tags: List[str]) -> List[Dict]:
        """Search Hugging Face for models matching given tags."""
        base_url = "https://huggingface.co/api/models"
        models = []
        
        for tag in tags:
            try:
                params = {
                    "search": tag,
                    "sort": "downloads",
                    "direction": "-1",
                    "limit": 100
                }
                
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    results = response.json()
                    for model in results:
                        if not any(existing["id"] == model["id"] for existing in models):
                            model["category"] = category
                            model["matched_tag"] = tag
                            models.append(model)
                
                time.sleep(1)  # Be nice to the API
                
            except Exception as e:
                logger.error(f"Error searching for tag {tag}: {e}")
        
        return models

    def research_papers(self, category: str, keywords: List[str]) -> List[Dict]:
        """Search for relevant research papers on arXiv."""
        base_url = "http://export.arxiv.org/api/query"
        papers = []
        
        search_query = " OR ".join(f'"{kw}"' for kw in keywords)
        try:
            params = {
                "search_query": f"all:{search_query}",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": 50
            }
            
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                # Parse XML response (simplified for now)
                papers = [{"title": "Paper parsing to be implemented"}]
            
            time.sleep(3)  # Be extra nice to arXiv
            
        except Exception as e:
            logger.error(f"Error searching papers for {category}: {e}")
        
        return papers

    def analyze_model(self, model_info: Dict) -> Dict:
        """Analyze a model's metadata for relevant information."""
        analysis = {
            "id": model_info.get("id", ""),
            "name": model_info.get("modelId", "").split("/")[-1],
            "category": model_info.get("category", ""),
            "downloads": model_info.get("downloads", 0),
            "likes": model_info.get("likes", 0),
            "tags": model_info.get("tags", []),
            "pipeline_tag": model_info.get("pipeline_tag", ""),
            "has_spaces": bool(model_info.get("spaces", [])),
            "last_modified": model_info.get("lastModified", ""),
            "matched_tag": model_info.get("matched_tag", ""),
            "size_info": self._extract_size_info(model_info)
        }
        return analysis

    def _extract_size_info(self, model_info: Dict) -> str:
        """Extract and format model size information."""
        try:
            siblings = model_info.get("siblings", [])
            total_size = sum(s.get("size", 0) for s in siblings if s.get("rfilename", "").endswith((".bin", ".safetensors")))
            return f"{total_size / (1024**3):.1f}GB" if total_size > 0 else "Unknown"
        except:
            return "Unknown"

    def generate_report(self, models_by_category: Dict[str, List[Dict]]) -> str:
        """Generate a detailed report of the research findings."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"model_research_report_{timestamp}.md")
        
        with open(report_file, "w") as f:
            f.write("# AI Model Research Report for Movie Making\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for category, models in models_by_category.items():
                f.write(f"## {category.replace('_', ' ').title()}\n\n")
                
                if not models:
                    f.write("No models found for this category.\n\n")
                    continue
                
                # Sort models by downloads
                models.sort(key=lambda x: x.get("downloads", 0), reverse=True)
                
                # Take top 10 models
                top_models = models[:10]
                
                f.write("| Model | Size | Downloads | Tags | Last Modified |\n")
                f.write("|-------|------|-----------|------|---------------|\n")
                
                for model in top_models:
                    name = model.get("id", "").split("/")[-1]
                    size = model.get("size_info", "Unknown")
                    downloads = f"{model.get('downloads', 0):,}"
                    tags = ", ".join(model.get("tags", [])[:3])  # Show first 3 tags
                    last_modified = model.get("last_modified", "").split("T")[0]
                    
                    f.write(f"| {name} | {size} | {downloads} | {tags} | {last_modified} |\n")
                
                f.write("\n")
        
        return report_file

    def save_results(self, models_by_category: Dict[str, List[Dict]]):
        """Save the raw research results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"model_research_results_{timestamp}.json")
        
        with open(results_file, "w") as f:
            json.dump(models_by_category, f, indent=2)
        
        logger.info(f"Saved raw results to {results_file}")

    def run_research(self):
        """Run the complete research process."""
        logger.info("Starting model research...")
        
        models_by_category = {}
        
        for category, tags in self.categories.items():
            logger.info(f"Researching {category}...")
            
            # Search Hugging Face
            models = self.search_huggingface(category, tags)
            
            # Analyze each model
            analyzed_models = [self.analyze_model(model) for model in models]
            
            models_by_category[category] = analyzed_models
            
            logger.info(f"Found {len(analyzed_models)} models for {category}")
            time.sleep(2)  # Pause between categories
        
        # Generate and save report
        report_file = self.generate_report(models_by_category)
        logger.info(f"Generated report: {report_file}")
        
        # Save raw results
        self.save_results(models_by_category)
        
        return report_file

def main():
    researcher = ModelResearcher()
    report_file = researcher.run_research()
    print(f"\nResearch complete! Report saved to: {report_file}")

if __name__ == "__main__":
    main() 