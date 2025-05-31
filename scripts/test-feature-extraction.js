require('dotenv').config();
const DataClient = require('../src/data/DataClient');
const FeatureExtractor = require('../src/data/FeatureExtractor');
const { Logger } = require('../src/utils');
const config = require('config');

async function testFeatureExtraction() {
    console.log('🚀 Testing Feature Extraction...');
    
    const dataClient = new DataClient();
    const featureExtractor = new FeatureExtractor(config.get('ml.features'));
    
    try {
        // Test 1: Get data from core
        console.log('\n📊 Test 1: Fetching data from core...');
        const rvnData = await dataClient.getPairData('RVN');
        console.log('✅ RVN data fetched:', {
            dataPoints: rvnData.history?.closes?.length || 0,
            indicators: Object.keys(rvnData.strategies || {}).length
        });
        
        // Test 2: Extract features
        console.log('\n📊 Test 2: Extracting features...');
        const result = featureExtractor.extractFeatures(rvnData);
        
        console.log('✅ Feature extraction successful!');
        console.log('Feature summary:', {
            totalFeatures: result.features.length,
            featureNames: result.featureNames.length,
            metadata: result.metadata
        });
        
        // Test 3: Display feature breakdown
        console.log('\n📊 Test 3: Feature breakdown...');
        
        // Group features by type
        const featureGroups = {};
        result.featureNames.forEach((name, index) => {
            const [group] = name.split('_');
            if (!featureGroups[group]) {
                featureGroups[group] = [];
            }
            featureGroups[group].push({
                name: name,
                value: result.features[index]
            });
        });
        
        Object.entries(featureGroups).forEach(([group, features]) => {
            console.log(`${group.toUpperCase()}: ${features.length} features`);
            // Show first few features as examples
            features.slice(0, 3).forEach(f => {
                console.log(`  ${f.name}: ${f.value.toFixed(4)}`);
            });
            if (features.length > 3) {
                console.log(`  ... and ${features.length - 3} more`);
            }
        });
        
        // Test 4: Create training targets
        console.log('\n📊 Test 4: Creating training targets...');
        const targets = featureExtractor.createTargets(rvnData.history);
        
        Object.entries(targets).forEach(([targetName, values]) => {
            console.log(`${targetName}: ${values.length} target values`);
            if (values.length > 0) {
                const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
                console.log(`  Average: ${avg.toFixed(4)}`);
            }
        });
        
        console.log('\n🎉 All feature extraction tests passed!');
        console.log(`Ready for ML training with ${result.features.length} features per sample`);
        
    } catch (error) {
        console.error('\n❌ Feature extraction test failed:', error.message);
        Logger.error('Feature extraction test failed', { error: error.message });
        process.exit(1);
    }
}

testFeatureExtraction();