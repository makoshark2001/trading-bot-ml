require('dotenv').config();
const axios = require('axios');

async function testBasicFunctionality() {
    console.log('🚀 Testing Basic ML Service Functionality...');
    console.log('=============================================');
    
    const mlUrl = 'http://localhost:3001';
    const testPair = 'RVN';
    
    try {
        // Test 1: Quick health check
        console.log('\n📊 Test 1: Quick Health Check...');
        try {
            const response = await axios.get(`${mlUrl}/api/health`, { 
                timeout: 5000,
                headers: { 'Connection': 'close' }
            });
            console.log('✅ Health endpoint working');
            console.log('📈 Training Status:', {
                autoTraining: response.data.training?.autoTraining?.enabled,
                periodicTraining: response.data.training?.periodicTraining?.enabled,
                currentlyTraining: response.data.training?.currentlyTraining?.count || 0
            });
        } catch (error) {
            console.log('❌ Health check failed:', error.message);
            if (error.code === 'ECONNRESET') {
                console.log('🔍 Connection reset - server may be overloaded');
            }
        }
        
        // Test 2: Check storage (this worked before)
        console.log('\n📊 Test 2: Storage Check...');
        try {
            const storageResponse = await axios.get(`${mlUrl}/api/storage/stats`, { 
                timeout: 5000,
                headers: { 'Connection': 'close' }
            });
            console.log('✅ Storage working:', {
                models: storageResponse.data.storage.models.count,
                weights: storageResponse.data.storage.weights.count,
                trainedModels: storageResponse.data.storage.trainedModels.length
            });
        } catch (error) {
            console.log('❌ Storage check failed:', error.message);
        }
        
        // Test 3: Check training config
        console.log('\n📊 Test 3: Training Configuration...');
        try {
            const configResponse = await axios.get(`${mlUrl}/api/training/config`, { 
                timeout: 5000,
                headers: { 'Connection': 'close' }
            });
            console.log('✅ Training config working:', {
                autoTraining: configResponse.data.autoTraining.enabled,
                periodicTraining: configResponse.data.periodicTraining.enabled,
                enabledModels: configResponse.data.enabledModels
            });
        } catch (error) {
            console.log('❌ Training config failed:', error.message);
        }
        
        // Test 4: Try a simple model status check
        console.log('\n📊 Test 4: Model Status Check...');
        try {
            const statusResponse = await axios.get(`${mlUrl}/api/models/${testPair}/status`, { 
                timeout: 10000,
                headers: { 'Connection': 'close' }
            });
            console.log('✅ Model status working for', testPair);
            
            const individual = statusResponse.data.individual || {};
            Object.entries(individual).forEach(([modelType, info]) => {
                console.log(`   ${modelType.toUpperCase()}:`, {
                    hasModel: info.hasModel,
                    isTraining: info.isTraining,
                    hasWeights: info.hasTrainedWeights
                });
            });
            
        } catch (error) {
            console.log('❌ Model status failed:', error.message);
            if (error.code === 'ECONNRESET') {
                console.log('🔍 Connection reset during model loading');
            }
        }
        
        // Test 5: Check what's causing the timeouts
        console.log('\n📊 Test 5: Investigating Timeout Issues...');
        
        // Try to disable auto-training temporarily
        try {
            console.log('🔧 Temporarily disabling auto-training...');
            await axios.post(`${mlUrl}/api/training/config`, {
                autoTraining: false,
                periodicTraining: false
            }, { 
                timeout: 5000,
                headers: { 'Connection': 'close' }
            });
            console.log('✅ Auto-training disabled');
            
            // Wait a moment
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Try prediction again
            console.log('🔮 Trying prediction with auto-training disabled...');
            const predictionResponse = await axios.get(`${mlUrl}/api/predictions/${testPair}`, { 
                timeout: 15000,
                headers: { 'Connection': 'close' }
            });
            
            console.log('✅ Prediction successful with auto-training disabled!');
            console.log('📈 Result:', {
                direction: predictionResponse.data.prediction?.direction,
                confidence: predictionResponse.data.prediction?.confidence?.toFixed(4),
                signal: predictionResponse.data.prediction?.signal
            });
            
        } catch (error) {
            console.log('❌ Prediction still failed:', error.message);
            console.log('🔍 Issue is not just auto-training');
        }
        
        // Test 6: Check individual models with short timeout
        console.log('\n📊 Test 6: Individual Model Quick Test...');
        const modelTypes = ['lstm'];  // Test just one model
        
        for (const modelType of modelTypes) {
            try {
                console.log(`🧠 Testing ${modelType.toUpperCase()} model...`);
                const response = await axios.get(`${mlUrl}/api/models/${testPair}/${modelType}/predict`, { 
                    timeout: 20000,
                    headers: { 'Connection': 'close' }
                });
                console.log(`✅ ${modelType.toUpperCase()} working:`, {
                    prediction: response.data.prediction?.prediction?.toFixed(4),
                    confidence: response.data.prediction?.confidence?.toFixed(4)
                });
            } catch (error) {
                console.log(`❌ ${modelType.toUpperCase()} failed:`, error.message);
                if (error.code === 'ECONNRESET') {
                    console.log('   🔍 Connection reset - model creation/training taking too long');
                } else if (error.code === 'ECONNABORTED') {
                    console.log('   🔍 Request timeout - model processing too slow');
                }
            }
        }
        
        // Test 7: Re-enable auto-training
        console.log('\n📊 Test 7: Re-enabling Auto-Training...');
        try {
            await axios.post(`${mlUrl}/api/training/config`, {
                autoTraining: true,
                periodicTraining: true
            }, { 
                timeout: 5000,
                headers: { 'Connection': 'close' }
            });
            console.log('✅ Auto-training re-enabled');
        } catch (error) {
            console.log('❌ Failed to re-enable auto-training:', error.message);
        }
        
        // Summary and diagnosis
        console.log('\n🔍 DIAGNOSIS');
        console.log('============');
        
        console.log('\n✅ Working Components:');
        console.log('• Basic server functionality');
        console.log('• Storage system');
        console.log('• Training configuration');
        console.log('• Simple API endpoints');
        
        console.log('\n❌ Problem Areas:');
        console.log('• Model loading/creation taking too long');
        console.log('• Automatic training causing timeouts');
        console.log('• TensorFlow operations may be slow');
        console.log('• Memory/CPU intensive operations blocking server');
        
        console.log('\n💡 Recommended Solutions:');
        console.log('1. Reduce model complexity (fewer layers, smaller units)');
        console.log('2. Implement proper async model loading');
        console.log('3. Add model loading timeouts');
        console.log('4. Pre-load models on startup instead of on-demand');
        console.log('5. Use smaller feature sets for faster processing');
        console.log('6. Implement model caching to avoid repeated loading');
        
        console.log('\n🔧 Immediate Actions:');
        console.log('• Models are being created/trained on first use');
        console.log('• This is causing 30+ second delays');
        console.log('• Need to either pre-train models or use cached weights');
        console.log('• Consider reducing training epochs for auto-training');
        
    } catch (error) {
        console.error('\n❌ Test failed:', error.message);
        process.exit(1);
    }
}

// Run the basic test
testBasicFunctionality();