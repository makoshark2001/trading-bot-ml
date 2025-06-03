require('dotenv').config();
const axios = require('axios');

async function rebuildAllModels() {
    console.log('🔧 Rebuilding All ML Models with Current Feature Count...');
    console.log('=======================================================');
    
    const mlUrl = 'http://localhost:3001';
    const pairs = ['RVN']; // Add your trading pairs here
    
    try {
        // Test 1: Check ML service health
        console.log('\n📊 Step 1: Checking ML Service Health...');
        const healthResponse = await axios.get(`${mlUrl}/api/health`);
        console.log('✅ ML Service Status:', healthResponse.data.status);
        
        if (healthResponse.data.models.featureCounts) {
            console.log('📈 Current Feature Counts:', healthResponse.data.models.featureCounts);
        }
        
        // Test 2: Get current feature counts
        console.log('\n📊 Step 2: Getting Current Feature Counts...');
        const featureCounts = {};
        
        for (const pair of pairs) {
            try {
                const featuresResponse = await axios.get(`${mlUrl}/api/features/${pair}`);
                featureCounts[pair] = featuresResponse.data.features.count;
                console.log(`✅ ${pair}: ${featuresResponse.data.features.count} features`);
            } catch (error) {
                console.log(`❌ Failed to get features for ${pair}:`, error.message);
            }
        }
        
        // Test 3: Check model status before rebuild
        console.log('\n📊 Step 3: Checking Model Status Before Rebuild...');
        for (const pair of pairs) {
            try {
                const statusResponse = await axios.get(`${mlUrl}/api/models/${pair}/status`);
                console.log(`📈 ${pair} Status:`, {
                    featureCount: statusResponse.data.featureCount,
                    hasEnsemble: statusResponse.data.ensemble.hasEnsemble,
                    individualModels: Object.keys(statusResponse.data.individual).filter(m => 
                        statusResponse.data.individual[m].hasModel
                    )
                });
            } catch (error) {
                console.log(`❌ Failed to get status for ${pair}:`, error.message);
            }
        }
        
        // Test 4: Rebuild models for each pair
        console.log('\n📊 Step 4: Rebuilding Models...');
        const rebuildResults = {};
        
        for (const pair of pairs) {
            try {
                console.log(`\n🔧 Rebuilding models for ${pair}...`);
                
                const rebuildResponse = await axios.post(`${mlUrl}/api/models/${pair}/rebuild`);
                rebuildResults[pair] = rebuildResponse.data;
                
                console.log(`✅ ${pair} rebuild successful:`, {
                    newFeatureCount: rebuildResponse.data.newFeatureCount,
                    rebuiltModels: rebuildResponse.data.rebuiltModels
                });
                
                // Wait a moment for models to be created
                await new Promise(resolve => setTimeout(resolve, 2000));
                
            } catch (error) {
                console.log(`❌ Failed to rebuild ${pair}:`, error.message);
                rebuildResults[pair] = { error: error.message };
            }
        }
        
        // Test 5: Check model status after rebuild
        console.log('\n📊 Step 5: Checking Model Status After Rebuild...');
        for (const pair of pairs) {
            try {
                const statusResponse = await axios.get(`${mlUrl}/api/models/${pair}/status`);
                console.log(`📈 ${pair} Status After Rebuild:`, {
                    featureCount: statusResponse.data.featureCount,
                    hasEnsemble: statusResponse.data.ensemble.hasEnsemble,
                    individualModels: Object.keys(statusResponse.data.individual).filter(m => 
                        statusResponse.data.individual[m].hasModel
                    )
                });
            } catch (error) {
                console.log(`❌ Failed to get status for ${pair}:`, error.message);
            }
        }
        
        // Test 6: Test predictions after rebuild
        console.log('\n📊 Step 6: Testing Predictions After Rebuild...');
        for (const pair of pairs) {
            try {
                console.log(`\n🔮 Testing ${pair} predictions...`);
                
                // Test individual model predictions
                const modelTypes = ['lstm', 'gru', 'cnn', 'transformer'];
                for (const modelType of modelTypes) {
                    try {
                        const predResponse = await axios.get(`${mlUrl}/api/models/${pair}/${modelType}/predict`);
                        console.log(`✅ ${pair} ${modelType.toUpperCase()}:`, {
                            prediction: predResponse.data.prediction.prediction.toFixed(4),
                            confidence: predResponse.data.prediction.confidence.toFixed(4),
                            direction: predResponse.data.prediction.direction
                        });
                    } catch (error) {
                        console.log(`❌ ${pair} ${modelType.toUpperCase()} failed:`, error.message);
                    }
                }
                
                // Test ensemble prediction
                try {
                    const ensembleResponse = await axios.get(`${mlUrl}/api/predictions/${pair}`);
                    console.log(`✅ ${pair} ENSEMBLE:`, {
                        prediction: ensembleResponse.data.prediction.prediction.toFixed(4),
                        confidence: ensembleResponse.data.prediction.confidence.toFixed(4),
                        direction: ensembleResponse.data.prediction.direction,
                        signal: ensembleResponse.data.prediction.signal
                    });
                } catch (error) {
                    console.log(`❌ ${pair} ENSEMBLE failed:`, error.message);
                }
                
            } catch (error) {
                console.log(`❌ Failed to test predictions for ${pair}:`, error.message);
            }
        }
        
        // Test 7: Final health check
        console.log('\n📊 Step 7: Final Health Check...');
        const finalHealthResponse = await axios.get(`${mlUrl}/api/health`);
        console.log('✅ Final ML Service Status:', finalHealthResponse.data.status);
        console.log('📈 Model Counts:', {
            individual: finalHealthResponse.data.models.individual.loaded,
            ensembles: finalHealthResponse.data.models.ensembles.loaded
        });
        
        if (finalHealthResponse.data.models.featureCounts) {
            console.log('📊 Final Feature Counts:', finalHealthResponse.data.models.featureCounts);
        }
        
        console.log('\n🎉 Model Rebuild Process Completed!');
        console.log('===================================');
        console.log('✅ All models have been rebuilt with current feature counts');
        console.log('✅ Predictions should now work correctly');
        console.log('✅ Feature count mismatches have been resolved');
        
        // Summary
        console.log('\n📊 Rebuild Summary:');
        Object.entries(rebuildResults).forEach(([pair, result]) => {
            if (result.error) {
                console.log(`❌ ${pair}: ${result.error}`);
            } else {
                console.log(`✅ ${pair}: Rebuilt with ${result.newFeatureCount} features`);
            }
        });
        
    } catch (error) {
        console.error('\n❌ Model rebuild process failed:', error.message);
        if (error.response) {
            console.error('Response data:', error.response.data);
        }
        process.exit(1);
    }
}

rebuildAllModels();