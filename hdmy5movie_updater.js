/**
 * HDMY 5 Movie - Video Gallery Updater
 * This script checks for new videos and updates the HTML gallery
 */

// Configuration
const config = {
    progressFile: '/home/tdeshane/movie_maker/hdmy5movie_videos/progress.json',
    videoBaseDir: '/home/tdeshane/movie_maker/hdmy5movie_videos',
    htmlFile: '/home/tdeshane/movie_maker/hdmy5movie.html',
    updateInterval: 60000, // Check for updates every minute
    sectionMapping: {
        '01_opening_credits': 'opening-credits-grid',
        '02_prologue': 'prologue-grid',
        '03_act1': 'act1-grid',
        '04_interlude1': 'interlude1-grid',
        '05_act2': 'act2-grid',
        '06_interlude2': 'interlude2-grid',
        '07_act3': 'act3-grid',
        '08_epilogue': 'epilogue-grid',
        '09_credits': 'credits-grid'
    }
};

// Utility functions
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

/**
 * Read the progress file to get current generation status
 */
function readProgress() {
    try {
        const data = fs.readFileSync(config.progressFile, 'utf8');
        return JSON.parse(data);
    } catch (error) {
        console.error('Error reading progress file:', error);
        return {
            total: 400,
            current: 0,
            percentage: 0,
            last_updated: new Date().toISOString(),
            current_section: '',
            current_prompt: ''
        };
    }
}

/**
 * Get all video files from the output directories
 */
function getVideoFiles() {
    const videos = {};
    
    // For each section directory
    Object.keys(config.sectionMapping).forEach(section => {
        const sectionDir = path.join(config.videoBaseDir, section);
        
        try {
            if (fs.existsSync(sectionDir)) {
                const files = fs.readdirSync(sectionDir)
                    .filter(file => file.endsWith('.mp4'))
                    .map(file => ({
                        path: path.join(section, file),
                        name: file,
                        fullPath: path.join(sectionDir, file),
                        created: fs.statSync(path.join(sectionDir, file)).mtime.getTime()
                    }))
                    .sort((a, b) => {
                        // Extract the first number from the filename (e.g., "001_002.mp4" -> 1)
                        const aNum = parseInt(a.name.split('_')[0]);
                        const bNum = parseInt(b.name.split('_')[0]);
                        return aNum - bNum;
                    });
                
                videos[section] = files;
            }
        } catch (error) {
            console.error(`Error reading directory ${sectionDir}:`, error);
            videos[section] = [];
        }
    });
    
    return videos;
}

/**
 * Generate HTML for a video item
 */
function generateVideoHTML(video, prompt) {
    // Extract sequence numbers from filename (e.g., "001_002.mp4")
    const parts = path.basename(video.name, '.mp4').split('_');
    const overallNum = parseInt(parts[0]);
    const sectionNum = parseInt(parts[1]);
    
    // Generate a title based on the sequence numbers
    const title = `Scene ${overallNum} (Section ${sectionNum})`;
    
    // Use the first 100 characters of the prompt as the description
    const description = prompt ? 
        (prompt.length > 100 ? prompt.substring(0, 100) + '...' : prompt) : 
        'Generated video scene';
    
    return `
    <div class="video-item">
        <div class="video-container">
            <video controls>
                <source src="hdmy5movie_videos/${video.path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <div class="video-info">
            <h3 class="video-title">${title}</h3>
            <p class="video-description">${description}</p>
        </div>
    </div>
    `;
}

/**
 * Update the HTML file with the latest videos
 */
function updateHTML(videos, progress) {
    try {
        // Read the HTML file
        let html = fs.readFileSync(config.htmlFile, 'utf8');
        
        // Update the progress bar
        const progressPercentage = progress.percentage || 0;
        html = html.replace(
            /(const percentage = Math\.floor\(\(generated \/ totalClips\) \* 100\);)/g,
            `const percentage = ${progressPercentage};`
        );
        
        html = html.replace(
            /(progressText\.textContent = `\${generated}\/\${totalClips} clips generated \(\${percentage}%\)`;)/g,
            `progressText.textContent = '${progress.current}/${progress.total} clips generated (${progressPercentage}%)';`
        );
        
        // Update each section with videos
        Object.keys(config.sectionMapping).forEach(section => {
            const sectionVideos = videos[section] || [];
            const sectionId = config.sectionMapping[section];
            
            if (sectionVideos.length > 0) {
                // Generate HTML for all videos in this section
                const videoHTML = sectionVideos.map(video => {
                    // Try to find the prompt for this video
                    let prompt = '';
                    if (progress && progress.current_prompt && 
                        video.name.includes(progress.current_section)) {
                        prompt = progress.current_prompt;
                    }
                    return generateVideoHTML(video, prompt);
                }).join('\n');
                
                // Replace the section content
                const sectionRegex = new RegExp(`(<div class="video-grid" id="${sectionId}">)[\\s\\S]*?(</div>\\s*<h2|</div>\\s*</div>\\s*<footer)`, 'g');
                html = html.replace(sectionRegex, `$1\n${videoHTML}\n$2`);
            }
        });
        
        // Write the updated HTML
        fs.writeFileSync(config.htmlFile, html);
        console.log(`Updated HTML file at ${new Date().toISOString()}`);
    } catch (error) {
        console.error('Error updating HTML:', error);
    }
}

/**
 * Main update function
 */
function update() {
    console.log('Checking for updates...');
    
    // Read the current progress
    const progress = readProgress();
    
    // Get all video files
    const videos = getVideoFiles();
    
    // Update the HTML
    updateHTML(videos, progress);
}

// Run the update function immediately
update();

// Then set up interval to check periodically
setInterval(update, config.updateInterval);

console.log(`HDMY 5 Movie updater started. Checking for updates every ${config.updateInterval / 1000} seconds.`); 