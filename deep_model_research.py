#!/usr/bin/env python3
"""
Deep Model Research Script for Movie Making

This script performs in-depth research on the top models from each category
and creates a prioritized download list.
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deep_model_research.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("deep_model_research")

# Define the top models to research in-depth
TOP_MODELS = {
    "video_generation": [
        "cerspense/zeroscope_v2_XL",
        "damo-vilab/text-to-video-ms-1.7b",
        "stabilityai/stable-video-diffusion-img2vid-xt",
        "runwayml/stable-video-diffusion-1-open",
        "ByteDance/AnimateDiff",
    ],
    "image_generation": [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        "SG161222/RealVisXL_V4.0",
        "stabilityai/stable-cascade",
        "stabilityai/sdxl-turbo",
    ],
    "audio_processing": [
        "suno/bark",
        "suno/audio-lm",
        "facebook/musicgen-stereo-large",
        "cvssp/audioldm-s-full",
        "openai/whisper-large-v3",
    ],
    "multimodal": [
        "microsoft/Phi-4-multimodal-instruct",
        "microsoft/Florence-2-large-no-flash-attn",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "HuggingFaceM4/Idefics3-8B-Llama3",
        "THUDM/cogvlm2-llama3-chat-19B",
    ],
    "language_models": [
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-70B",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
        "microsoft/Phi-3-mini-4k-instruct",
    ]
}

class DeepModelResearcher:
    def __init__(self):
        self.results_dir = "model_research"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load previous research if available
        self.prev_research_file = self._find_latest_research_file()
        self.previous_research = self._load_previous_research()
        
    def _find_latest_research_file(self) -> Optional[str]:
        """Find the most recent research results file."""
        result_files = [
            f for f in os.listdir(self.results_dir) 
            if f.startswith("model_research_results_") and f.endswith(".json")
        ]
        
        if not result_files:
            return None
            
        # Sort by timestamp in filename
        result_files.sort(reverse=True)
        return os.path.join(self.results_dir, result_files[0])
    
    def _load_previous_research(self) -> Dict:
        """Load previous research results if available."""
        if not self.prev_research_file or not os.path.exists(self.prev_research_file):
            logger.info("No previous research found, starting fresh.")
            return {}
            
        try:
            with open(self.prev_research_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded previous research from {self.prev_research_file}")
                return data
        except Exception as e:
            logger.error(f"Error loading previous research: {e}")
            return {}

    def get_model_details(self, model_id: str) -> Dict:
        """Get detailed information about a specific model from Hugging Face API."""
        try:
            # Check if we already have this model in previous research
            for category, models in self.previous_research.items():
                for model in models:
                    if model.get("id") == model_id:
                        logger.info(f"Found {model_id} in previous research")
                        return model
                
            # If not found, fetch from API
            url = f"https://huggingface.co/api/models/{model_id}"
            response = requests.get(url)
            
            if response.status_code == 200:
                model_data = response.json()
                logger.info(f"Retrieved details for {model_id}")
                return model_data
            else:
                logger.error(f"Error retrieving {model_id}: {response.status_code}")
                return {"id": model_id, "error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Exception when retrieving model {model_id}: {e}")
            return {"id": model_id, "error": str(e)}

    def analyze_model_deeply(self, model_data: Dict) -> Dict:
        """Perform deep analysis of a model's capabilities and requirements."""
        if "error" in model_data:
            return model_data
            
        model_id = model_data.get("id", "unknown")
        
        # Calculate size in GB
        siblings = model_data.get("siblings", [])
        # Count only model files (safetensors, bin, ckpt) and exclude optimized models (GGUF, GGML, etc.)
        model_files = [
            s for s in siblings 
            if s.get("rfilename", "").endswith((".bin", ".safetensors", ".ckpt", ".pt", ".pth"))
            and not any(x in s.get("rfilename", "").lower() for x in ["gguf", "ggml", "q4", "q5", "q8"])
        ]
        total_size = sum(s.get("size", 0) for s in model_files)
        size_gb = total_size / (1024**3) if total_size > 0 else 0
        
        # Determine framework
        tags = model_data.get("tags", [])
        framework = "unknown"
        if "pytorch" in tags:
            framework = "pytorch"
        elif "tensorflow" in tags or "tf" in tags:
            framework = "tensorflow"
        elif "jax" in tags or "flax" in tags:
            framework = "jax"
        
        # Check for quantized versions
        has_quantized = any("quantized" in s.get("rfilename", "").lower() 
                            or "gguf" in s.get("rfilename", "").lower()
                            or "ggml" in s.get("rfilename", "").lower()
                            or "q4" in s.get("rfilename", "").lower()
                            or "q5" in s.get("rfilename", "").lower()
                            or "q8" in s.get("rfilename", "").lower()
                            for s in siblings)
        
        # Check for license issues
        license_tag = next((tag for tag in tags if tag.startswith("license:")), None)
        if license_tag:
            license_type = license_tag.split(":")[-1]
        else:
            license_type = model_data.get("cardData", {}).get("license", "unknown")
        
        # Determine hardware requirements based on size
        hardware_req = "CPU only"
        if size_gb > 1:
            hardware_req = "GPU recommended"
        if size_gb > 10:
            hardware_req = "GPU required"
        if size_gb > 30:
            hardware_req = "High-end GPU required"
        
        return {
            "id": model_id,
            "name": model_id.split("/")[-1],
            "full_name": model_id,
            "size_gb": round(size_gb, 2),
            "downloads": model_data.get("downloads", 0),
            "likes": model_data.get("likes", 0),
            "tags": model_data.get("tags", []),
            "framework": framework,
            "has_quantized": has_quantized,
            "last_modified": model_data.get("lastModified", ""),
            "license": license_type,
            "hardware_req": hardware_req,
            "pipeline_tag": model_data.get("pipeline_tag", ""),
            "tasks": model_data.get("pipeline_tag", "").split("-") if model_data.get("pipeline_tag") else [],
        }
        
    def prioritize_models(self, analyzed_models: List[Dict]) -> List[Dict]:
        """Prioritize models based on various factors."""
        if not analyzed_models:
            return []
            
        for model in analyzed_models:
            # Calculate a priority score (lower is better)
            # Factors: popularity, size, license restrictions, hardware requirements
            popularity_score = 1 / (model.get("downloads", 1) + 1) * 1000
            size_penalty = model.get("size_gb", 0) * 0.5  # Smaller models preferred
            
            # License penalty
            license_penalty = 0
            license_type = model.get("license", "unknown").lower()
            if "cc-by-nc" in license_type or "non-commercial" in license_type:
                license_penalty = 20
            elif "proprietary" in license_type or "gated" in license_type:
                license_penalty = 50
                
            # Hardware penalty
            hw_penalty = 0
            hw_req = model.get("hardware_req", "").lower()
            if "gpu required" in hw_req:
                hw_penalty = 10
            if "high-end gpu required" in hw_req:
                hw_penalty = 30
                
            # Quantized bonus
            quantized_bonus = -15 if model.get("has_quantized", False) else 0
                
            # Calculate final score
            priority_score = popularity_score + size_penalty + license_penalty + hw_penalty + quantized_bonus
            
            model["priority_score"] = round(priority_score, 2)
        
        # Sort by priority score (ascending)
        return sorted(analyzed_models, key=lambda x: x.get("priority_score", float('inf')))
        
    def generate_download_list(self, prioritized_models: Dict[str, List[Dict]]) -> str:
        """Generate a download script for the prioritized models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_script = os.path.join(self.results_dir, f"download_models_{timestamp}.sh")
        
        with open(download_script, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Model Download Script for Movie Making\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("# Create directories for different model types\n")
            f.write("mkdir -p models/video_generation\n")
            f.write("mkdir -p models/image_generation\n")
            f.write("mkdir -p models/audio_processing\n")
            f.write("mkdir -p models/multimodal\n")
            f.write("mkdir -p models/language_models\n\n")
            
            f.write("# Use Hugging Face CLI to download models\n")
            f.write("# You may need to log in first with: huggingface-cli login\n")
            f.write("# Token can be generated at: https://huggingface.co/settings/tokens\n\n")
            
            f.write('if [ -z "$HF_TOKEN" ]; then\n')
            f.write('  echo "HF_TOKEN environment variable not set. Some models may not download."\n')
            f.write('  echo "To set: export HF_TOKEN=your_token_here"\n')
            f.write('fi\n\n')
            
            total_size = 0
            for category, models in prioritized_models.items():
                f.write(f"# {category.replace('_', ' ').title()} Models\n")
                f.write(f"echo \"\\nDownloading {category.replace('_', ' ')} models...\"\n")
                
                for i, model in enumerate(models[:3]):  # Top 3 from each category
                    model_id = model.get("full_name", "")
                    size_gb = model.get("size_gb", 0)
                    total_size += size_gb
                    
                    f.write(f"\n# {model_id} ({size_gb:.2f}GB)\n")
                    f.write(f"echo \"Downloading {model_id} ({size_gb:.2f}GB)...\"\n")
                    
                    # Different download methods based on model type
                    if "diffusers" in model.get("tags", []):
                        f.write(f"python -c \"from diffusers import DiffusionPipeline; "
                                f"DiffusionPipeline.from_pretrained('{model_id}', "
                                f"use_auth_token=\\\"$HF_TOKEN\\\", cache_dir='models/{category}')\"\n")
                    else:
                        f.write(f"python -c \"from transformers import AutoModel; "
                                f"AutoModel.from_pretrained('{model_id}', "
                                f"use_auth_token=\\\"$HF_TOKEN\\\", cache_dir='models/{category}')\"\n")
                
                f.write("\n")
            
            f.write(f"\necho \"All downloads complete! Total size: {total_size:.2f}GB\"\n")
            
        # Make executable
        os.chmod(download_script, 0o755)
        return download_script
        
    def generate_report(self, analyzed_models_by_category: Dict[str, List[Dict]]) -> str:
        """Generate a detailed report of the deeper research findings."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"deep_model_research_report_{timestamp}.md")
        
        with open(report_file, "w") as f:
            f.write("# Deep AI Model Research Report for Movie Making\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary section with total model size
            total_size = sum(
                model.get("size_gb", 0) 
                for category_models in analyzed_models_by_category.values() 
                for model in category_models[:3]  # Top 3 per category
            )
            f.write(f"## Summary\n\n")
            f.write(f"This report contains detailed analysis of top models for movie making.\n")
            f.write(f"Total download size for recommended models (top 3 per category): **{total_size:.2f}GB**\n\n")
            
            for category, models in analyzed_models_by_category.items():
                category_name = category.replace('_', ' ').title()
                f.write(f"## {category_name}\n\n")
                
                if not models:
                    f.write("No models found for this category.\n\n")
                    continue
                
                # Sort models by priority_score
                models.sort(key=lambda x: x.get("priority_score", float('inf')))
                
                category_size = sum(model.get("size_gb", 0) for model in models[:3])
                f.write(f"**Top 3 models total size: {category_size:.2f}GB**\n\n")
                
                f.write("| Priority | Model | Size | Downloads | License | Hardware Req | Tags |\n")
                f.write("|----------|-------|------|-----------|---------|--------------|------|\n")
                
                for i, model in enumerate(models, 1):
                    name = model.get("name", "")
                    full_name = model.get("full_name", "")
                    size = f"{model.get('size_gb', 0):.2f}GB"
                    downloads = f"{model.get('downloads', 0):,}"
                    license_type = model.get("license", "unknown")
                    hw_req = model.get("hardware_req", "Unknown")
                    tags = ", ".join(model.get("tags", [])[:3])  # Show first 3 tags
                    
                    priority = i
                    priority_label = "⭐" * (4 - min(i, 3)) if i <= 5 else ""
                    
                    f.write(f"| {priority} {priority_label} | [{name}](https://huggingface.co/{full_name}) | {size} | {downloads} | {license_type} | {hw_req} | {tags} |\n")
                
                f.write("\n")
                
                # Add detailed information for top 3 models
                f.write("### Detailed Analysis\n\n")
                for i, model in enumerate(models[:3], 1):
                    name = model.get("name", "")
                    full_name = model.get("full_name", "")
                    
                    f.write(f"#### {i}. {name}\n\n")
                    f.write(f"- **Full name:** `{full_name}`\n")
                    f.write(f"- **Size:** {model.get('size_gb', 0):.2f}GB\n")
                    f.write(f"- **Downloads:** {model.get('downloads', 0):,}\n")
                    f.write(f"- **License:** {model.get('license', 'unknown')}\n")
                    f.write(f"- **Framework:** {model.get('framework', 'unknown')}\n")
                    f.write(f"- **Hardware requirements:** {model.get('hardware_req', 'Unknown')}\n")
                    f.write(f"- **Has quantized versions:** {'Yes' if model.get('has_quantized', False) else 'No'}\n")
                    f.write(f"- **Last modified:** {model.get('last_modified', '').split('T')[0]}\n")
                    f.write(f"- **Pipeline tag:** {model.get('pipeline_tag', 'unknown')}\n")
                    f.write(f"- **Priority score:** {model.get('priority_score', 0)}\n")
                    
                    # Download command
                    if "diffusers" in model.get("tags", []):
                        download_cmd = (f"python -c \"from diffusers import DiffusionPipeline; "
                                      f"DiffusionPipeline.from_pretrained('{full_name}', use_auth_token='YOUR_HF_TOKEN')\"")
                    else:
                        download_cmd = (f"python -c \"from transformers import AutoModel; "
                                      f"AutoModel.from_pretrained('{full_name}', use_auth_token='YOUR_HF_TOKEN')\"")
                    
                    f.write(f"- **Download command:**\n```python\n{download_cmd}\n```\n\n")
                
                f.write("\n")
        
        return report_file

    def run_deep_research(self):
        """Run the deep research process on top models."""
        logger.info("Starting deep model research...")
        
        analyzed_models_by_category = {}
        
        for category, model_list in TOP_MODELS.items():
            logger.info(f"Researching {category} models deeply...")
            analyzed_models = []
            
            for model_id in tqdm(model_list, desc=f"Researching {category}"):
                # Get detailed model information
                model_data = self.get_model_details(model_id)
                
                # Analyze model deeply
                analyzed_model = self.analyze_model_deeply(model_data)
                analyzed_models.append(analyzed_model)
                
                # Be nice to the API
                time.sleep(1)
            
            # Prioritize models in this category
            prioritized_models = self.prioritize_models(analyzed_models)
            analyzed_models_by_category[category] = prioritized_models
            
            logger.info(f"Completed deep research for {len(prioritized_models)} {category} models")
        
        # Generate detailed report
        report_file = self.generate_report(analyzed_models_by_category)
        logger.info(f"Generated detailed report: {report_file}")
        
        # Generate download script
        download_script = self.generate_download_list(analyzed_models_by_category)
        logger.info(f"Generated download script: {download_script}")
        
        return report_file, download_script

def main():
    researcher = DeepModelResearcher()
    report_file, download_script = researcher.run_deep_research()
    print(f"\nDeep research complete!")
    print(f"- Detailed report: {report_file}")
    print(f"- Download script: {download_script}")
    print("\nTo download models, run:")
    print(f"chmod +x {download_script}")
    print(f"HF_TOKEN=your_huggingface_token {download_script}")

if __name__ == "__main__":
    main() 