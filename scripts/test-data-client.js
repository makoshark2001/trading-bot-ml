require('dotenv').config();
const DataClient = require('../src/data/DataClient');
const { Logger } = require('../src/utils');

async function testDataClient() {
    console.log('🚀 Testing ML Data Client...');
    
    const dataClient = new DataClient();
    
    try {
        // Test 1: Health check
        console.log('\n📊 Test 1: Core Service Health Check');
        const health = await dataClient.checkCoreHealth();
        console.log('✅ Core service status:', health.status);
        console.log('Data points:', health.dataCollection?.totalDataPoints || 0);
        
        // Test 2: Get all data
        console.log('\n📊 Test 2: Fetch All Data');
        const allData = await dataClient.getAllData();
        console.log('✅ Pairs available:', allData.pairs?.length || 0);
        console.log('Pairs:', allData.pairs);
        
        // Test 3: Get specific pair data
        console.log('\n📊 Test 3: Fetch RVN Data');
        const rvnData = await dataClient.getPairData('RVN');
        console.log('✅ RVN data points:', rvnData.history?.closes?.length || 0);
        console.log('RVN strategies available:', Object.keys(rvnData.strategies || {}));
        
        console.log('\n🎉 All ML Data Client tests passed!');
        
    } catch (error) {
        console.error('\n❌ ML Data Client test failed:', error.message);
        Logger.error('ML Data Client test failed', { error: error.message });
        process.exit(1);
    }
}

testDataClient();