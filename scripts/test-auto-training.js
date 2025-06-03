require('dotenv').config();
const axios = require('axios');

async function testAutoAndPeriodicTraining() {
    console.log('🚀 Testing Automatic & Periodic Training Functionality...');
    console.log('========================================================');
    
    const mlUrl = 'http://localhost:3001';
    const testPair = 'RVN';
    
    try {
        // Test 1: Check if ML service is running
        console.log('\n📊 Test 1: Checking ML Service Availability...');
        try {
            const response = await axios.get(`${mlUrl}/api/health`, { timeout: 5000 });
            console.log('✅ ML Service is running on port 3001');
            console.log('📈 Service Status:', response.data.status);
            console.log('🔧 Service Type:', response.data.service || 'trading-bot-ml');
        } catch (error) {
            if (error.code === 'ECONNREFUSED') {
                console.log('❌ ML Service is not running on port 3001');
                console.log('💡 Please start the ML service with: npm start');
                console.log('🔍 Make sure the trading-bot-core service is also running on port 3000');
                return;
            }
            throw error;
        }
        
        // Test 2: Check detailed health with training features
        console.log('\n📊 Test 2: Checking Enhanced ML Service Health...');
        const healthResponse = await axios.get(`${mlUrl}/api/health`);
        console.log('✅ ML Service Status:', healthResponse.data.status);
        console.log('📈 Service Version:', healthResponse.data.service || 'trading-bot-ml');
        
        // Check if service has training features
        if (healthResponse.data.training) {
            console.log('🤖 Training Features Available:');
            console.log('   Auto Training:', healthResponse.data.training.autoTraining?.enabled || 'Unknown');
            console.log('   Periodic Training:', healthResponse.data.training.periodicTraining?.enabled || 'Unknown');
            console.log('   Currently Training:', healthResponse.data.training.currentlyTraining?.count || 0);
        } else {
            console.log('⚠️ Training features not detected in health response');
            console.log('🔍 Available properties:', Object.keys(healthResponse.data));
        }
        
        // Check core service connection
        if (healthResponse.data.core) {
            console.log('🔗 Core Service Connection:', healthResponse.data.core.status);
        } else {
            console.log('⚠️ Core service connection status not available');
        }
        
        // Test 3: Try to get basic prediction (this will test if service works)
        console.log('\n📊 Test 3: Testing Basic Prediction...');
        try {
            const predictionResponse = await axios.get(`${mlUrl}/api/predictions/${testPair}`, { timeout: 30000 });
            console.log('✅ Basic prediction successful!');
            console.log('📈 Prediction Details:');
            console.log('   Direction:', predictionResponse.data.prediction?.direction || 'Unknown');
            console.log('   Confidence:', predictionResponse.data.prediction?.confidence?.toFixed(4) || 'Unknown');
            console.log('   Signal:', predictionResponse.data.prediction?.signal || 'Unknown');
            console.log('   Ensemble:', predictionResponse.data.ensemble || false);
            
            if (predictionResponse.data.autoTraining) {
                console.log('🤖 Auto Training Status:');
                console.log('   Enabled:', predictionResponse.data.autoTraining.enabled);
                console.log('   Currently Training:', predictionResponse.data.autoTraining.currentlyTraining);
            }
        } catch (error) {
            console.log('❌ Basic prediction failed:', error.message);
            if (error.response) {
                console.log('   Status:', error.response.status);
                console.log('   Data:', error.response.data);
            }
            
            // If basic prediction fails, the service might not be properly configured
            console.log('🔍 This suggests the service may not be fully functional yet');
        }
        
        // Test 4: Check available endpoints
        console.log('\n📊 Test 4: Checking Available Endpoints...');
        const endpoints = [
            '/api/health',
            '/api/training/status',
            '/api/training/config',
            `/api/models/${testPair}/status`,
            '/api/storage/stats'
        ];
        
        for (const endpoint of endpoints) {
            try {
                const response = await axios.get(`${mlUrl}${endpoint}`, { timeout: 10000 });
                console.log(`✅ ${endpoint}: Available (${response.status})`);
                
                // Show some key info for important endpoints
                if (endpoint === '/api/training/status' && response.data.training) {
                    console.log('   Auto Training:', response.data.training.autoTraining?.enabled);
                    console.log('   Periodic Training:', response.data.training.periodicTraining?.enabled);
                    console.log('   Currently Training:', response.data.training.currentlyTraining?.count || 0);
                }
                
                if (endpoint === '/api/training/config') {
                    console.log('   Auto Training Config:', response.data.autoTraining?.enabled);
                    console.log('   Periodic Training Config:', response.data.periodicTraining?.enabled);
                    console.log('   Training Interval:', response.data.periodicTraining?.intervalHours, 'hours');
                }
                
            } catch (error) {
                console.log(`❌ ${endpoint}: Not available (${error.response?.status || error.message})`);
                if (error.response?.data) {
                    console.log('   Error:', error.response.data.error || error.response.data.message);
                }
            }
        }
        
        // Test 5: Check model status
        console.log('\n📊 Test 5: Checking Model Status...');
        try {
            const statusResponse = await axios.get(`${mlUrl}/api/models/${testPair}/status`);
            console.log('✅ Model status retrieved successfully');
            console.log('📚 Model Status for', testPair);
            
            if (statusResponse.data.individual) {
                Object.entries(statusResponse.data.individual).forEach(([modelType, info]) => {
                    console.log(`   ${modelType.toUpperCase()}:`, {
                        hasModel: info.hasModel || false,
                        hasTrainedWeights: info.hasTrainedWeights || false,
                        isTraining: info.isTraining || false
                    });
                });
            } else {
                console.log('   Individual model info not available');
            }
            
            if (statusResponse.data.training) {
                console.log('Training Configuration:');
                console.log('   Auto Training:', statusResponse.data.training.autoTraining);
                console.log('   Periodic Training:', statusResponse.data.training.periodicTraining);
                console.log('   Currently Training:', statusResponse.data.training.currentlyTraining);
            }
            
        } catch (error) {
            console.log('❌ Model status check failed:', error.message);
            if (error.response?.data) {
                console.log('   Error details:', error.response.data);
            }
        }
        
        // Test 6: Check storage
        console.log('\n📊 Test 6: Checking Storage System...');
        try {
            const storageResponse = await axios.get(`${mlUrl}/api/storage/stats`);
            console.log('✅ Storage system available');
            console.log('💾 Storage Statistics:');
            
            if (storageResponse.data.storage) {
                const storage = storageResponse.data.storage;
                console.log('   Models:', storage.models?.count || 0, 'files');
                console.log('   Weights:', storage.weights?.count || 0, 'directories');
                console.log('   Training History:', storage.training?.count || 0, 'files');
                console.log('   Predictions:', storage.predictions?.count || 0, 'files');
                console.log('   Total Size:', Math.round((storage.totalSizeBytes || 0) / 1024), 'KB');
                
                if (storage.trainedModels) {
                    console.log('   Trained Models:', storage.trainedModels.length);
                }
            } else {
                console.log('   Storage details not available in response');
            }
            
        } catch (error) {
            console.log('❌ Storage check failed:', error.message);
            if (error.response?.data) {
                console.log('   Error details:', error.response.data);
            }
        }
        
        // Test 7: Test individual model endpoints
        console.log('\n📊 Test 7: Testing Individual Model Endpoints...');
        const modelTypes = ['lstm', 'gru', 'cnn', 'transformer'];
        
        for (const modelType of modelTypes) {
            try {
                const response = await axios.get(`${mlUrl}/api/models/${testPair}/${modelType}/predict`, { timeout: 30000 });
                console.log(`✅ ${modelType.toUpperCase()} model:`, {
                    prediction: response.data.prediction?.prediction?.toFixed(4) || 'N/A',
                    confidence: response.data.prediction?.confidence?.toFixed(4) || 'N/A',
                    direction: response.data.prediction?.direction || 'N/A'
                });
            } catch (error) {
                console.log(`❌ ${modelType.toUpperCase()} model failed:`, error.response?.status || error.message);
                
                // If it's a 500 error, show more details
                if (error.response?.status === 500 && error.response?.data) {
                    console.log(`   Error: ${error.response.data.error || error.response.data.message}`);
                }
            }
        }
        
        // Test 8: Test configuration update (if available)
        console.log('\n📊 Test 8: Testing Configuration Update...');
        try {
            const configResponse = await axios.post(`${mlUrl}/api/training/config`, {
                autoTraining: true,
                periodicTraining: true
            });
            console.log('✅ Configuration update successful:', configResponse.data.message || 'Updated');
        } catch (error) {
            console.log('❌ Configuration update failed:', error.response?.status || error.message);
            if (error.response?.data) {
                console.log('   Error:', error.response.data.error || error.response.data.message);
            }
        }
        
        // Summary
        console.log('\n🎉 Service Test Summary');
        console.log('======================');
        console.log('✅ Service is running and accessible');
        console.log('📊 Basic functionality tested');
        
        // Check what's working and what's not
        const workingFeatures = [];
        const failingFeatures = [];
        
        try {
            await axios.get(`${mlUrl}/api/health`);
            workingFeatures.push('Health check');
        } catch (e) {
            failingFeatures.push('Health check');
        }
        
        try {
            await axios.get(`${mlUrl}/api/predictions/${testPair}`, { timeout: 15000 });
            workingFeatures.push('Predictions');
        } catch (e) {
            failingFeatures.push('Predictions');
        }
        
        try {
            await axios.get(`${mlUrl}/api/training/status`);
            workingFeatures.push('Training status');
        } catch (e) {
            failingFeatures.push('Training status');
        }
        
        try {
            await axios.get(`${mlUrl}/api/storage/stats`);
            workingFeatures.push('Storage system');
        } catch (e) {
            failingFeatures.push('Storage system');
        }
        
        console.log('\n✅ Working Features:');
        workingFeatures.forEach(feature => console.log(`   • ${feature}`));
        
        if (failingFeatures.length > 0) {
            console.log('\n❌ Features with Issues:');
            failingFeatures.forEach(feature => console.log(`   • ${feature}`));
        }
        
        console.log('\n💡 Next Steps:');
        if (failingFeatures.includes('Predictions')) {
            console.log('• Fix prediction functionality (check core service connection)');
        }
        if (failingFeatures.includes('Training status')) {
            console.log('• Check if automatic training features are properly implemented');
        }
        if (failingFeatures.includes('Storage system')) {
            console.log('• Verify storage system is working correctly');
        }
        
        console.log('• Monitor logs for specific error messages');
        console.log('• Check that trading-bot-core service is running on port 3000');
        console.log('• Verify all required dependencies are installed');
        
    } catch (error) {
        console.error('\n❌ Service test failed:', error.message);
        
        if (error.code === 'ECONNREFUSED') {
            console.log('\n💡 Connection Refused - Service Not Running');
            console.log('Please make sure:');
            console.log('1. The ML service is started with: npm start');
            console.log('2. The service is running on port 3001');
            console.log('3. The trading-bot-core service is running on port 3000');
        } else if (error.response) {
            console.error('Response status:', error.response.status);
            console.error('Response data:', error.response.data);
        }
        
        console.log('\n🔍 Debugging Steps:');
        console.log('1. Check the service logs for startup errors');
        console.log('2. Verify all environment variables are set correctly');
        console.log('3. Make sure all dependencies are installed: npm install');
        console.log('4. Check if port 3001 is available: netstat -an | find "3001"');
        
        process.exit(1);
    }
}

// Run the test
console.log('Starting ML Service Test...');
console.log('==========================');
testAutoAndPeriodicTraining();