// scripts/debug-ensemble.js
require('dotenv').config();
const tf = require('@tensorflow/tfjs');
const LSTMModel = require('../src/models/LSTMModel');
const GRUModel = require('../src/models/GRUModel');
const CNNModel = require('../src/models/CNNModel');
const TransformerModel = require('../src/models/TransformerModel');
const ModelEnsemble = require('../src/models/ModelEnsemble');
const { Logger } = require('../src/utils');

async function debugEnsemble() {
    console.log('🔍 Debugging Ensemble Issues...');
    console.log('===============================');
    
    const sequenceLength = 10;
    const features = 5;
    const modelConfigs = {
        lstm: {
            sequenceLength: sequenceLength,
            features: features,
            units: 10,
            layers: 1,
            dropout: 0.1,
            learningRate: 0.001
        },
        gru: {
            sequenceLength: sequenceLength,
            features: features,
            units: 10,
            layers: 1,
            dropout: 0.1,
            learningRate: 0.001
        },
        cnn: {
            sequenceLength: sequenceLength,
            features: features,
            filters: [8, 16],
            kernelSizes: [3, 3],
            poolSizes: [2, 2],
            denseUnits: [32, 16],
            dropout: 0.2,
            learningRate: 0.001
        },
        transformer: {
            sequenceLength: sequenceLength,
            features: features,
            dModel: 32,
            numHeads: 2,
            numLayers: 2,
            dff: 64,
            dropout: 0.1,
            learningRate: 0.001
        }
    };
    
    const models = {};
    const buildResults = {};
    
    // Test each model individually
    for (const [modelType, config] of Object.entries(modelConfigs)) {
        console.log(`\n📊 Testing ${modelType.toUpperCase()} model...`);
        
        try {
            let model;
            
            switch (modelType) {
                case 'lstm':
                    model = new LSTMModel(config);
                    break;
                case 'gru':
                    model = new GRUModel(config);
                    break;
                case 'cnn':
                    model = new CNNModel(config);
                    break;
                case 'transformer':
                    model = new TransformerModel(config);
                    break;
            }
            
            // Test build
            console.log(`  🔧 Building ${modelType}...`);
            model.buildModel();
            console.log(`  ✅ ${modelType} built successfully`);
            
            // Test compile
            console.log(`  🔧 Compiling ${modelType}...`);
            model.compileModel();
            console.log(`  ✅ ${modelType} compiled successfully`);
            
            // Test prediction with dummy data
            console.log(`  🔧 Testing ${modelType} prediction...`);
            const dummyInput = tf.randomNormal([2, sequenceLength, features]);
            const predictions = await model.predict(dummyInput);
            console.log(`  ✅ ${modelType} prediction successful:`, Array.from(predictions).map(p => p.toFixed(4)));
            
            dummyInput.dispose();
            
            models[modelType] = model;
            buildResults[modelType] = { success: true, error: null };
            
        } catch (error) {
            console.log(`  ❌ ${modelType} failed:`, error.message);
            buildResults[modelType] = { success: false, error: error.message };
        }
    }
    
    console.log('\n📈 Build Results Summary:');
    console.log('========================');
    Object.entries(buildResults).forEach(([modelType, result]) => {
        if (result.success) {
            console.log(`✅ ${modelType.toUpperCase()}: Working`);
        } else {
            console.log(`❌ ${modelType.toUpperCase()}: Failed - ${result.error}`);
        }
    });
    
    const workingModels = Object.keys(models);
    console.log(`\n🚀 Working models: ${workingModels.length}/4`);
    console.log(`   Models: ${workingModels.join(', ')}`);
    
    if (workingModels.length >= 2) {
        console.log('\n📊 Testing ensemble with working models...');
        
        try {
            const ensemble = new ModelEnsemble({
                modelTypes: workingModels,
                votingStrategy: 'weighted'
            });
            
            // Add working models
            workingModels.forEach(modelType => {
                ensemble.addModel(modelType, models[modelType], 1.0 / workingModels.length, {
                    created: Date.now()
                });
            });
            
            // Test ensemble prediction
            const testInput = tf.randomNormal([1, sequenceLength, features]);
            const ensemblePrediction = await ensemble.predict(testInput);
            
            console.log('✅ Ensemble prediction successful:', {
                prediction: ensemblePrediction.prediction.toFixed(4),
                confidence: ensemblePrediction.confidence.toFixed(4),
                direction: ensemblePrediction.direction,
                signal: ensemblePrediction.signal,
                modelCount: ensemblePrediction.ensemble.modelCount
            });
            
            testInput.dispose();
            ensemble.dispose();
            
        } catch (error) {
            console.log('❌ Ensemble test failed:', error.message);
        }
    } else {
        console.log('❌ Not enough working models for ensemble (need at least 2)');
    }
    
    // Cleanup
    Object.values(models).forEach(model => {
        if (model && typeof model.dispose === 'function') {
            model.dispose();
        }
    });
    
    console.log('\n🎯 Debug completed!');
}

debugEnsemble().catch(error => {
    console.error('Debug failed:', error);
    process.exit(1);
});