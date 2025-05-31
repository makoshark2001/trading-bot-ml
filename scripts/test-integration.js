require('dotenv').config();
const axios = require('axios');

async function testIntegration() {
    console.log('🚀 Testing Core ↔ ML Integration...');
    
    const coreUrl = 'http://localhost:3000';
    const mlUrl = 'http://localhost:3001';
    
    try {
        // Test 1: Core service health
        console.log('\n📊 Test 1: Core service health...');
        const coreHealth = await axios.get(`${coreUrl}/api/health`);
        console.log('✅ Core status:', coreHealth.data.status);
        
        // Test 2: ML service health
        console.log('\n📊 Test 2: ML service health...');
        const mlHealth = await axios.get(`${mlUrl}/api/health`);
        console.log('✅ ML status:', mlHealth.data.status);
        console.log('Core connection:', mlHealth.data.core.status);
        
        // Test 3: ML predictions
        console.log('\n📊 Test 3: ML predictions...');
        const predictions = await axios.get(`${mlUrl}/api/predictions/RVN`);
        console.log('✅ RVN prediction:', {
            direction: predictions.data.prediction.direction,
            signal: predictions.data.prediction.signal,
            confidence: predictions.data.prediction.confidence.toFixed(3)
        });
        
        // Test 4: Feature extraction
        console.log('\n📊 Test 4: Feature extraction...');
        const features = await axios.get(`${mlUrl}/api/features/RVN`);
        console.log('✅ Features extracted:', {
            count: features.data.features.count,
            sample: features.data.features.values
        });
        
        console.log('\n🎉 Integration tests passed!');
        console.log('🔄 Core ↔ ML communication working perfectly! 🚀');
        
    } catch (error) {
        console.error('\n❌ Integration test failed:', error.message);
        if (error.response) {
            console.error('Response:', error.response.data);
        }
        process.exit(1);
    }
}

testIntegration();