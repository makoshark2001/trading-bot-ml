// File: scripts/clear-old-models.js
// Create this new file to clear old model metadata

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { Logger } = require('../src/utils');

async function clearOldModels() {
    console.log('🧹 Clearing old model metadata...');
    
    const baseDir = 'data/ml';
    const modelsDir = path.join(baseDir, 'models');
    
    try {
        if (!fs.existsSync(modelsDir)) {
            console.log('✅ No models directory found - nothing to clear');
            return;
        }
        
        const files = fs.readdirSync(modelsDir);
        let clearedCount = 0;
        
        console.log(`📁 Found ${files.length} files in models directory`);
        
        for (const file of files) {
            if (file.endsWith('.json')) {
                const filePath = path.join(modelsDir, file);
                
                try {
                    // Read the file to check if it's a model metadata file
                    const content = fs.readFileSync(filePath, 'utf8');
                    const data = JSON.parse(content);
                    
                    // Check if this is a model metadata file
                    if (data.modelInfo || data.ensembleConfig || data.type === 'model_metadata') {
                        console.log(`🗑️  Removing old model metadata: ${file}`);
                        fs.unlinkSync(filePath);
                        clearedCount++;
                    } else {
                        console.log(`⏭️  Skipping non-model file: ${file}`);
                    }
                } catch (error) {
                    console.log(`⚠️  Failed to process ${file}: ${error.message}`);
                }
            }
        }
        
        console.log(`✅ Cleared ${clearedCount} old model metadata files`);
        
        // Also clear any cached models in memory by restarting the service
        console.log('');
        console.log('🔄 Please restart the ML service to apply changes:');
        console.log('   npm start');
        console.log('');
        console.log('💡 This will force the service to create new models with the correct feature count (84)');
        
    } catch (error) {
        console.error('❌ Failed to clear old models:', error.message);
        Logger.error('Clear old models failed', { error: error.message });
        process.exit(1);
    }
}

clearOldModels();