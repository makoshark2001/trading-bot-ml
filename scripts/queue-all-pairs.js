// Script to queue training for all available pairs
// Save this as: scripts/queue-all-pairs.js

const axios = require('axios');

const ML_SERVICE_URL = 'http://localhost:3001';
const CORE_SERVICE_URL = 'http://localhost:3000';

// Configuration for training
const TRAINING_CONFIG = {
    epochs: 25,           // Moderate training (not too long)
    batchSize: 32,        // Good batch size
    priority: 3,          // Manual priority (higher than periodic)
    maxAttempts: 2,       // Retry once if failed
    source: 'bulk_queue', // Mark as bulk queuing
};

// Model types to train (all 4 models)
const MODEL_TYPES = ['lstm', 'gru', 'cnn', 'transformer'];

async function getAllAvailablePairs() {
    try {
        console.log('🔍 Getting all available pairs from core service...');
        
        // Method 1: Try to get pairs from core service config
        try {
            const response = await axios.get(`${CORE_SERVICE_URL}/api/config`, { timeout: 10000 });
            if (response.data.config && response.data.config.pairs) {
                console.log(`✅ Found ${response.data.config.pairs.length} pairs from core config`);
                return response.data.config.pairs.map(pair => pair.toUpperCase());
            }
        } catch (error) {
            console.log('⚠️ Could not get pairs from core config, trying alternative methods...');
        }

        // Method 2: Try to get pairs from available pairs endpoint
        try {
            const response = await axios.get(`${CORE_SERVICE_URL}/api/available-pairs`, { timeout: 10000 });
            if (response.data.availablePairs) {
                const activePairs = response.data.availablePairs
                    .filter(p => p.isActive)
                    .map(p => p.pair.toUpperCase());
                console.log(`✅ Found ${activePairs.length} active pairs from available pairs`);
                return activePairs;
            }
        } catch (error) {
            console.log('⚠️ Could not get pairs from available pairs endpoint...');
        }

        // Method 3: Use fallback pairs
        const fallbackPairs = ['BTC', 'ETH', 'XMR', 'RVN', 'LTC', 'ADA'];
        console.log(`📋 Using fallback pairs: ${fallbackPairs.join(', ')}`);
        return fallbackPairs;

    } catch (error) {
        console.error('❌ Failed to get available pairs:', error.message);
        return ['BTC', 'ETH', 'XMR']; // Minimal fallback
    }
}

async function checkMLServiceHealth() {
    try {
        console.log('🏥 Checking ML service health...');
        const response = await axios.get(`${ML_SERVICE_URL}/api/health`, { timeout: 5000 });
        
        console.log('✅ ML Service Status:', {
            status: response.data.status,
            enabledModels: response.data.models?.enabledTypes || [],
            queueActive: response.data.training?.queue?.active?.count || 0,
            queueQueued: response.data.training?.queue?.queued?.count || 0,
        });
        
        return response.data.status === 'healthy';
    } catch (error) {
        console.error('❌ ML service health check failed:', error.message);
        return false;
    }
}

async function checkTrainingAllowed(pair, modelType) {
    try {
        const response = await axios.get(`${ML_SERVICE_URL}/api/models/${pair}/status`, { timeout: 5000 });
        
        if (response.data.individual && response.data.individual[modelType]) {
            const training = response.data.individual[modelType].training;
            return {
                allowed: training.allowed,
                reason: training.reason,
                cooldownRemaining: training.cooldownRemainingMinutes
            };
        }
        
        return { allowed: true, reason: 'No restrictions found' };
    } catch (error) {
        console.log(`⚠️ Could not check training status for ${pair}:${modelType}, assuming allowed`);
        return { allowed: true, reason: 'Status check failed' };
    }
}

async function queueTrainingJob(pair, modelType, config) {
    try {
        console.log(`📝 Queuing ${pair}:${modelType}...`);
        
        const response = await axios.post(
            `${ML_SERVICE_URL}/api/train/${pair}/${modelType}`,
            config,
            { timeout: 10000 }
        );
        
        if (response.data.results && response.data.results.length > 0) {
            const result = response.data.results[0];
            if (result.status === 'queued') {
                console.log(`✅ ${pair}:${modelType} queued successfully (Job ID: ${result.jobId})`);
                return { success: true, jobId: result.jobId };
            } else {
                console.log(`⚠️ ${pair}:${modelType} queue failed: ${result.error || result.reason}`);
                return { success: false, error: result.error || result.reason };
            }
        }
        
        console.log(`❌ ${pair}:${modelType} unexpected response format`);
        return { success: false, error: 'Unexpected response format' };
        
    } catch (error) {
        console.log(`❌ ${pair}:${modelType} queue failed: ${error.message}`);
        return { success: false, error: error.message };
    }
}

async function queueAllPairs() {
    console.log('🚀 Starting bulk training queue for all pairs...\n');
    
    // Check ML service health first
    const isHealthy = await checkMLServiceHealth();
    if (!isHealthy) {
        console.error('❌ ML service is not healthy, aborting...');
        return;
    }
    
    // Get all available pairs
    const pairs = await getAllAvailablePairs();
    console.log(`\n📊 Found ${pairs.length} pairs to process: ${pairs.join(', ')}\n`);
    
    const results = {
        queued: 0,
        skipped: 0,
        failed: 0,
        total: 0,
        details: []
    };
    
    // Process each pair with all model types
    for (const pair of pairs) {
        console.log(`\n🔄 Processing ${pair}...`);
        
        for (const modelType of MODEL_TYPES) {
            results.total++;
            
            // Check if training is allowed
            const trainingCheck = await checkTrainingAllowed(pair, modelType);
            
            if (!trainingCheck.allowed) {
                console.log(`⏭️ Skipping ${pair}:${modelType} - ${trainingCheck.reason}`);
                if (trainingCheck.cooldownRemaining) {
                    console.log(`   Cooldown remaining: ${trainingCheck.cooldownRemaining} minutes`);
                }
                results.skipped++;
                results.details.push({
                    pair,
                    modelType,
                    status: 'skipped',
                    reason: trainingCheck.reason
                });
                continue;
            }
            
            // Queue the training job
            const queueResult = await queueTrainingJob(pair, modelType, TRAINING_CONFIG);
            
            if (queueResult.success) {
                results.queued++;
                results.details.push({
                    pair,
                    modelType,
                    status: 'queued',
                    jobId: queueResult.jobId
                });
            } else {
                results.failed++;
                results.details.push({
                    pair,
                    modelType,
                    status: 'failed',
                    error: queueResult.error
                });
            }
            
            // Small delay between requests to avoid overwhelming the service
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        // Longer delay between pairs
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    // Print summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 BULK TRAINING QUEUE SUMMARY');
    console.log('='.repeat(60));
    console.log(`✅ Queued: ${results.queued}`);
    console.log(`⏭️ Skipped: ${results.skipped}`);
    console.log(`❌ Failed: ${results.failed}`);
    console.log(`📊 Total: ${results.total}`);
    console.log('='.repeat(60));
    
    // Show queued jobs
    const queuedJobs = results.details.filter(d => d.status === 'queued');
    if (queuedJobs.length > 0) {
        console.log('\n✅ Successfully Queued Jobs:');
        queuedJobs.forEach(job => {
            console.log(`   ${job.pair}:${job.modelType} (Job ID: ${job.jobId})`);
        });
    }
    
    // Show failed jobs
    const failedJobs = results.details.filter(d => d.status === 'failed');
    if (failedJobs.length > 0) {
        console.log('\n❌ Failed Jobs:');
        failedJobs.forEach(job => {
            console.log(`   ${job.pair}:${job.modelType} - ${job.error}`);
        });
    }
    
    // Show skipped jobs
    const skippedJobs = results.details.filter(d => d.status === 'skipped');
    if (skippedJobs.length > 0) {
        console.log('\n⏭️ Skipped Jobs (Cooldowns):');
        skippedJobs.forEach(job => {
            console.log(`   ${job.pair}:${job.modelType} - ${job.reason}`);
        });
    }
    
    console.log('\n🎯 Bulk queue operation completed!');
    console.log('💡 Check training queue status: curl http://localhost:3001/api/training/queue');
}

// Allow running directly or as module
if (require.main === module) {
    queueAllPairs().catch(error => {
        console.error('❌ Bulk queue operation failed:', error.message);
        process.exit(1);
    });
}

module.exports = { queueAllPairs, getAllAvailablePairs };