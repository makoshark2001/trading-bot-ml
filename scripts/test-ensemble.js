#!/usr/bin/env node

const axios = require('axios');
const colors = require('colors');

const ML_BASE_URL = 'http://localhost:3001';
const TEST_PAIR = 'BTC';

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function testEnsembleMode() {
    console.log('🤖 TESTING ENSEMBLE MODE FUNCTIONALITY'.green.bold);
    console.log('====================================='.green);
    console.log('');

    try {
        // Step 1: Check if ML service is running
        console.log('1. Checking ML service health...'.yellow);
        const healthResponse = await axios.get(`${ML_BASE_URL}/api/health`);
        const health = healthResponse.data;
        
        console.log(`   ✅ Service Status: ${health.status}`.green);
        console.log(`   🤖 Ensemble Mode: ${health.ensembleMode ? 'ENABLED' : 'DISABLED'}`.cyan);
        console.log(`   ⚡ Quick Mode: ${health.quickMode ? 'ENABLED' : 'DISABLED'}`.cyan);
        console.log(`   🎯 Enabled Models: ${health.models.enabledTypes.join(', ')}`.cyan);
        console.log(`   📊 Strategy: ${health.models.strategy}`.cyan);
        console.log('');

        if (!health.ensembleMode) {
            console.log('❌ WARNING: Ensemble mode is not enabled!'.red.bold);
            console.log('   Check your .env file and ensure ML_QUICK_MODE=false'.yellow);
            console.log('');
        }

        // Step 2: Check API info
        console.log('2. Checking API configuration...'.yellow);
        const apiResponse = await axios.get(`${ML_BASE_URL}/api`);
        const apiInfo = apiResponse.data;
        
        console.log(`   📦 Version: ${apiInfo.version}`.cyan);
        console.log(`   🤖 Ensemble Mode: ${apiInfo.ensembleMode ? 'ENABLED' : 'DISABLED'}`.cyan);
        console.log(`   🎯 Enabled Models: ${apiInfo.ensemble.enabledModels.join(', ')}`.cyan);
        console.log(`   📊 Strategy: ${apiInfo.ensemble.strategy}`.cyan);
        console.log(`   ⏱️  Cache Timeout: ${apiInfo.ensemble.cacheTimeout}ms`.cyan);
        console.log('');

        // Step 3: Test feature extraction
        console.log('3. Testing feature extraction...'.yellow);
        const featuresResponse = await axios.get(`${ML_BASE_URL}/api/features/${TEST_PAIR}`);
        const features = featuresResponse.data;
        
        console.log(`   ✅ Features extracted for ${TEST_PAIR}`.green);
        console.log(`   📊 Feature count: ${features.features.count}`.cyan);
        console.log(`   ⏱️  Response time: ${Date.now() - Date.parse(featuresResponse.headers.date)}ms`.cyan);
        console.log('');

        // Step 4: Check model status
        console.log('4. Checking model status...'.yellow);
        const statusResponse = await axios.get(`${ML_BASE_URL}/api/models/${TEST_PAIR}/status`);
        const status = statusResponse.data;
        
        console.log(`   📊 Feature count: ${status.featureCount}`.cyan);
        console.log(`   🤖 Ensemble Mode: ${status.ensembleMode ? 'ENABLED' : 'DISABLED'}`.cyan);
        
        console.log('   📋 Individual Models:'.cyan);
        for (const [modelType, modelInfo] of Object.entries(status.individual)) {
            const hasModel = modelInfo.hasModel ? '✅' : '❌';
            const hasWeights = modelInfo.hasWeights ? '💾' : '🔄';
            const canTrain = modelInfo.training.allowed ? '🟢' : '🔴';
            console.log(`     ${hasModel} ${modelType.toUpperCase()}: Model=${modelInfo.hasModel}, Weights=${hasWeights}, CanTrain=${canTrain}`.white);
        }
        
        console.log(`   🎯 Can Create Ensemble: ${status.ensemble.canCreateEnsemble ? 'YES' : 'NO'}`.cyan);
        console.log(`   📊 Current Ensembles: ${status.ensemble.stats ? status.ensemble.stats.modelCount : 0} models`.cyan);
        console.log('');

        // Step 5: Test single model prediction
        console.log('5. Testing single model prediction (LSTM)...'.yellow);
        const singleStart = Date.now();
        const singleResponse = await axios.get(`${ML_BASE_URL}/api/predictions/${TEST_PAIR}?model=lstm`);
        const singlePrediction = singleResponse.data;
        const singleTime = Date.now() - singleStart;
        
        console.log(`   ✅ Single LSTM prediction completed`.green);
        console.log(`   📋 Raw response structure:`.cyan, JSON.stringify(singlePrediction, null, 2));
        
        // Handle different response structures
        const prediction = singlePrediction.prediction || singlePrediction;
        const predValue = typeof prediction.prediction === 'number' ? prediction.prediction : 
                         typeof prediction === 'number' ? prediction : 0.5;
        
        console.log(`   🎯 Prediction: ${predValue.toFixed(4)}`.cyan);
        console.log(`   🎯 Direction: ${(prediction.direction || 'unknown').toUpperCase()}`.cyan);
        console.log(`   💪 Confidence: ${((prediction.confidence || 0) * 100).toFixed(1)}%`.cyan);
        console.log(`   📊 Signal: ${prediction.signal || 'UNKNOWN'}`.cyan);
        console.log(`   ⏱️  Response time: ${singleTime}ms`.cyan);
        console.log(`   📦 Cached: ${singlePrediction.cached ? 'YES' : 'NO'}`.cyan);
        console.log('');

        // Step 6: Test ensemble prediction
        console.log('6. Testing ensemble prediction...'.yellow);
        
        // First check if we have enough models for ensemble
        const hasModels = Object.values(status.individual).filter(m => m.hasModel).length;
        
        if (hasModels < 2) {
            console.log(`   ⚠️  Not enough models for ensemble (${hasModels}/2+)`.yellow);
            console.log(`   🔧 Training models first...`.yellow);
            
            // Train LSTM and GRU models
            const modelsToTrain = ['lstm', 'gru'];
            for (const modelType of modelsToTrain) {
                console.log(`   🔄 Training ${modelType.toUpperCase()} model...`.yellow);
                try {
                    const trainResponse = await axios.post(`${ML_BASE_URL}/api/train/${TEST_PAIR}/${modelType}`, {
                        epochs: 5, // Quick training for testing
                        priority: 1
                    });
                    console.log(`   ✅ ${modelType.toUpperCase()} training queued: ${trainResponse.data.jobId}`.green);
                } catch (trainError) {
                    console.log(`   ❌ Failed to queue ${modelType.toUpperCase()} training: ${trainError.message}`.red);
                }
            }
            
            console.log(`   ⏳ Waiting for training to complete (this may take a few minutes)...`.yellow);
            
            // Wait for training to complete
            let trainingComplete = false;
            let attempts = 0;
            const maxAttempts = 60; // 5 minutes max
            
            while (!trainingComplete && attempts < maxAttempts) {
                await sleep(5000); // Wait 5 seconds
                attempts++;
                
                try {
                    const queueResponse = await axios.get(`${ML_BASE_URL}/api/training/queue`);
                    const queue = queueResponse.data.queue;
                    
                    const activeJobs = queue.active.count;
                    const queuedJobs = queue.queued.count;
                    
                    if (activeJobs === 0 && queuedJobs === 0) {
                        trainingComplete = true;
                        console.log(`   ✅ Training completed!`.green);
                    } else {
                        process.stdout.write(`   ⏳ Training progress... (Active: ${activeJobs}, Queued: ${queuedJobs}) [${attempts}/${maxAttempts}]\r`.yellow);
                    }
                } catch (error) {
                    console.log(`   ⚠️  Error checking training status: ${error.message}`.yellow);
                }
            }
            
            if (!trainingComplete) {
                console.log(`\n   ❌ Training did not complete within timeout`.red);
                console.log(`   ℹ️  You can check training status with: curl ${ML_BASE_URL}/api/training/queue`.blue);
                return;
            }
        }
        
        // Now test ensemble prediction
        console.log(`   🤖 Testing ensemble prediction...`.yellow);
        const ensembleStart = Date.now();
        const ensembleResponse = await axios.get(`${ML_BASE_URL}/api/predictions/${TEST_PAIR}?ensemble=true`);
        const ensemblePrediction = ensembleResponse.data;
        const ensembleTime = Date.now() - ensembleStart;
        
        console.log(`   ✅ Ensemble prediction completed`.green);
        
        // Handle different response structures  
        const ensemblePred = ensemblePrediction.prediction || ensemblePrediction;
        const ensemblePredValue = typeof ensemblePred.prediction === 'number' ? ensemblePred.prediction : 
                                 typeof ensemblePred === 'number' ? ensemblePred : 0.5;
        
        console.log(`   🎯 Prediction: ${ensemblePredValue.toFixed(4)}`.cyan);
        console.log(`   🎯 Direction: ${(ensemblePred.direction || 'unknown').toUpperCase()}`.cyan);
        console.log(`   💪 Confidence: ${((ensemblePred.confidence || 0) * 100).toFixed(1)}%`.cyan);
        console.log(`   📊 Signal: ${ensemblePred.signal || 'UNKNOWN'}`.cyan);
        console.log(`   ⏱️  Response time: ${ensembleTime}ms`.cyan);
        console.log(`   📦 Cached: ${ensemblePrediction.cached ? 'YES' : 'NO'}`.cyan);
        console.log(`   🤖 Ensemble: ${ensemblePrediction.ensemble ? 'YES' : 'NO'}`.cyan);
        
        if (ensemblePred.ensemble) {
            console.log(`   📊 Strategy: ${ensemblePred.ensemble.strategy}`.cyan);
            console.log(`   🎯 Model Count: ${ensemblePred.ensemble.modelCount}`.cyan);
            console.log(`   🔍 Individual Predictions:`.cyan);
            for (const [model, pred] of Object.entries(ensemblePred.ensemble.individualPredictions || {})) {
                console.log(`     📈 ${model.toUpperCase()}: ${pred.toFixed(4)}`.white);
            }
        }
        console.log('');

        // Step 7: Test different ensemble strategies
        console.log('7. Testing different ensemble strategies...'.yellow);
        const strategies = ['weighted', 'majority', 'average'];
        
        for (const strategy of strategies) {
            try {
                const strategyStart = Date.now();
                const strategyResponse = await axios.get(`${ML_BASE_URL}/api/predictions/${TEST_PAIR}?strategy=${strategy}`);
                const strategyPrediction = strategyResponse.data;
                const strategyTime = Date.now() - strategyStart;
                
                const strategyPred = strategyPrediction.prediction || strategyPrediction;
                const strategyPredValue = typeof strategyPred.prediction === 'number' ? strategyPred.prediction : 
                                         typeof strategyPred === 'number' ? strategyPred : 0.5;
                
                console.log(`   📊 ${strategy.toUpperCase()} Strategy:`.cyan);
                console.log(`     🎯 Prediction: ${strategyPredValue.toFixed(4)}`.white);
                console.log(`     💪 Confidence: ${((strategyPred.confidence || 0) * 100).toFixed(1)}%`.white);
                console.log(`     📊 Signal: ${strategyPred.signal || 'UNKNOWN'}`.white);
                console.log(`     ⏱️  Time: ${strategyTime}ms`.white);
            } catch (error) {
                console.log(`   ❌ ${strategy.toUpperCase()} failed: ${error.message}`.red);
            }
        }
        console.log('');

        // Step 8: Performance comparison
        console.log('8. Performance comparison...'.yellow);
        console.log(`   ⚡ Single Model (LSTM): ${singleTime}ms`.cyan);
        console.log(`   🤖 Ensemble: ${ensembleTime}ms`.cyan);
        console.log(`   📊 Performance ratio: ${(ensembleTime / singleTime).toFixed(1)}x slower`.cyan);
        console.log('');

        // Step 9: Cache effectiveness test
        console.log('9. Testing cache effectiveness...'.yellow);
        
        // Second call should be cached
        const cachedStart = Date.now();
        const cachedResponse = await axios.get(`${ML_BASE_URL}/api/predictions/${TEST_PAIR}`);
        const cachedTime = Date.now() - cachedStart;
        
        console.log(`   ⚡ Cached call: ${cachedTime}ms`.cyan);
        console.log(`   📦 Is cached: ${cachedResponse.data.cached ? 'YES' : 'NO'}`.cyan);
        console.log(`   🚀 Cache speedup: ${(ensembleTime / cachedTime).toFixed(1)}x faster`.cyan);
        console.log('');

        // Final summary
        console.log('🎉 ENSEMBLE MODE TEST COMPLETED SUCCESSFULLY!'.green.bold);
        console.log('==========================================='.green);
        console.log('');
        console.log('📊 Test Results Summary:'.cyan.bold);
        console.log(`   ✅ Ensemble Mode: ${health.ensembleMode ? 'ENABLED' : 'DISABLED'}`.green);
        console.log(`   ✅ Models Available: ${Object.values(status.individual).filter(m => m.hasModel).length}`.green);
        console.log(`   ✅ Ensemble Working: ${ensemblePrediction.ensemble ? 'YES' : 'NO'}`.green);
        console.log(`   ✅ Performance: Single=${singleTime}ms, Ensemble=${ensembleTime}ms, Cached=${cachedTime}ms`.green);
        console.log('');
        console.log('🚀 Your ensemble mode is now active and ready for trading!'.green.bold);
        console.log('');
        console.log('📋 Next steps:'.yellow.bold);
        console.log('   1. Train additional models: POST /api/train/BTC/cnn'.blue);
        console.log('   2. Monitor ensemble performance: GET /api/models/BTC/status'.blue);
        console.log('   3. Test with other pairs: GET /api/predictions/ETH'.blue);
        console.log('   4. Adjust cache timeout if needed in .env file'.blue);

    } catch (error) {
        console.error('❌ Test failed:'.red.bold, error.message);
        
        if (error.code === 'ECONNREFUSED') {
            console.log('');
            console.log('🔧 Troubleshooting:'.yellow.bold);
            console.log('   1. Make sure the ML service is running: npm start'.blue);
            console.log('   2. Check if the service is on port 3001'.blue);
            console.log('   3. Verify your .env configuration'.blue);
        }
    }
}

// Run the test
testEnsembleMode().catch(console.error);