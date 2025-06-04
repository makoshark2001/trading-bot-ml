#!/usr/bin/env node

const axios = require('axios');

const CORE_URL = 'http://localhost:3000';
const ML_URL = 'http://localhost:3001';

async function getAllTradingPairs() {
    console.log('🔍 Finding all trading pairs...');
    
    try {
        // Method 1: Get from core service configuration
        const coreConfig = await axios.get(`${CORE_URL}/api/config`);
        if (coreConfig.data.config && coreConfig.data.config.pairs) {
            console.log(`✅ Found pairs from core config: ${coreConfig.data.config.pairs.join(', ')}`);
            return coreConfig.data.config.pairs;
        }
    } catch (error) {
        console.log(`⚠️  Could not get pairs from core config: ${error.message}`);
    }
    
    try {
        // Method 2: Get current pairs from core service
        const corePairs = await axios.get(`${CORE_URL}/api/pairs`);
        if (corePairs.data.pairs) {
            console.log(`✅ Found pairs from core API: ${corePairs.data.pairs.join(', ')}`);
            return corePairs.data.pairs;
        }
    } catch (error) {
        console.log(`⚠️  Could not get pairs from core API: ${error.message}`);
    }
    
    try {
        // Method 3: Check what pairs ML service has seen
        const mlHealth = await axios.get(`${ML_URL}/api/health`);
        if (mlHealth.data.models && mlHealth.data.models.individual && mlHealth.data.models.individual.pairs) {
            const mlPairs = mlHealth.data.models.individual.pairs;
            console.log(`✅ Found pairs from ML service: ${mlPairs.join(', ')}`);
            return mlPairs;
        }
    } catch (error) {
        console.log(`⚠️  Could not get pairs from ML service: ${error.message}`);
    }
    
    // Fallback to common pairs
    console.log(`🔄 Using fallback common pairs`);
    return ['BTC', 'ETH', 'XMR', 'RVN', 'LTC', 'ADA', 'DOT', 'LINK'];
}

async function trainAllPairs() {
    console.log('🚀 TRAINING ALL TRADING PAIRS');
    console.log('============================');
    
    try {
        // Get all trading pairs
        const pairs = await getAllTradingPairs();
        console.log(`\n📊 Will train models for ${pairs.length} pairs: ${pairs.join(', ')}\n`);
        
        const trainingConfig = {
            epochs: 25,        // Quick training for all pairs
            priority: 1,       // High priority
            batchSize: 32,
            maxAttempts: 2
        };
        
        console.log(`⚙️  Training configuration:`, trainingConfig);
        console.log('');
        
        const results = [];
        
        // Queue training for each pair
        for (let i = 0; i < pairs.length; i++) {
            const pair = pairs[i];
            console.log(`🔄 [${i + 1}/${pairs.length}] Queuing training for ${pair}...`);
            
            try {
                const response = await axios.post(`${ML_URL}/api/train/${pair}`, trainingConfig);
                const result = response.data;
                
                if (result.results) {
                    // Multiple models trained
                    const successful = result.results.filter(r => r.status === 'queued').length;
                    const failed = result.results.filter(r => r.error).length;
                    
                    console.log(`   ✅ ${successful} models queued, ${failed} failed`);
                    results.push({
                        pair,
                        status: 'success',
                        modelsQueued: successful,
                        modelsFailed: failed,
                        details: result.results
                    });
                } else {
                    console.log(`   ✅ Training queued successfully`);
                    results.push({
                        pair,
                        status: 'success',
                        jobId: result.jobId
                    });
                }
                
            } catch (error) {
                console.log(`   ❌ Failed: ${error.message}`);
                results.push({
                    pair,
                    status: 'failed',
                    error: error.message
                });
            }
            
            // Small delay to avoid overwhelming the queue
            if (i < pairs.length - 1) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        
        // Summary
        console.log('\n📊 TRAINING QUEUE SUMMARY');
        console.log('========================');
        
        const successful = results.filter(r => r.status === 'success').length;
        const failed = results.filter(r => r.status === 'failed').length;
        
        console.log(`✅ Successfully queued: ${successful} pairs`);
        console.log(`❌ Failed to queue: ${failed} pairs`);
        
        if (failed > 0) {
            console.log('\n❌ Failed pairs:');
            results.filter(r => r.status === 'failed').forEach(r => {
                console.log(`   ${r.pair}: ${r.error}`);
            });
        }
        
        // Show training queue status
        console.log('\n🔄 Current Training Queue Status:');
        try {
            const queueResponse = await axios.get(`${ML_URL}/api/training/queue`);
            const queue = queueResponse.data.queue;
            
            console.log(`   Active jobs: ${queue.active.count}`);
            console.log(`   Queued jobs: ${queue.queued.count}`);
            console.log(`   Total jobs: ${queue.active.count + queue.queued.count}`);
            
            if (queue.active.jobs.length > 0) {
                console.log('\n   🔄 Currently training:');
                queue.active.jobs.forEach(job => {
                    const duration = Math.round((Date.now() - job.startedAt) / 1000);
                    console.log(`     ${job.pair}:${job.modelType} (${duration}s)`);
                });
            }
            
            if (queue.queued.jobs.length > 0) {
                console.log('\n   ⏳ In queue:');
                queue.queued.jobs.slice(0, 10).forEach((job, index) => {
                    console.log(`     ${index + 1}. ${job.pair}:${job.modelType} (priority: ${job.priority})`);
                });
                
                if (queue.queued.jobs.length > 10) {
                    console.log(`     ... and ${queue.queued.jobs.length - 10} more`);
                }
            }
            
        } catch (error) {
            console.log(`   ⚠️  Could not get queue status: ${error.message}`);
        }
        
        console.log('\n🎯 Next Steps:');
        console.log('   1. Monitor training progress: curl http://localhost:3001/api/training/queue');
        console.log('   2. Check model status: curl http://localhost:3001/api/models/BTC/status');
        console.log('   3. Test ensemble predictions once training completes');
        console.log('\n⏳ Training will take approximately 5-15 minutes per model depending on your hardware.');
        
    } catch (error) {
        console.error('❌ Script failed:', error.message);
        process.exit(1);
    }
}

// Run the script
trainAllPairs().catch(console.error);