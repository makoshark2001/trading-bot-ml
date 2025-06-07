// Script to check TensorFlow backend and optimization
// Save this as: scripts/check-tensorflow-backend.js

const tf = require('@tensorflow/tfjs');

async function checkTensorFlowBackend() {
    console.log('🔍 Checking TensorFlow.js Backend Configuration...\n');
    
    try {
        // Wait for TensorFlow to be ready
        await tf.ready();
        
        // Get basic info
        console.log('📊 TensorFlow.js Information:');
        console.log('='.repeat(50));
        console.log(`Version: ${tf.version.tfjs}`);
        console.log(`Current Backend: ${tf.getBackend()}`);
        console.log(`Platform: ${tf.env().platform}`);
        console.log(`Features: ${JSON.stringify(tf.env().features, null, 2)}`);
        
        // Check available backends
        console.log('\n🔧 Available Backends:');
        console.log('='.repeat(50));
        const backends = tf.engine().registryFactory;
        Object.keys(backends).forEach(backend => {
            console.log(`- ${backend}`);
        });
        
        // Check memory info
        console.log('\n💾 Memory Information:');
        console.log('='.repeat(50));
        const memInfo = tf.memory();
        console.log(`Number of Tensors: ${memInfo.numTensors}`);
        console.log(`Number of Bytes: ${memInfo.numBytes}`);
        console.log(`Unreliable: ${memInfo.unreliable || false}`);
        
        // Test tensor operations performance
        console.log('\n⚡ Performance Test:');
        console.log('='.repeat(50));
        
        const sizes = [100, 500, 1000];
        for (const size of sizes) {
            const startTime = Date.now();
            
            // Create test tensors
            const a = tf.randomNormal([size, size]);
            const b = tf.randomNormal([size, size]);
            
            // Perform matrix multiplication
            const result = tf.matMul(a, b);
            await result.data(); // Force execution
            
            const endTime = Date.now();
            const duration = endTime - startTime;
            
            // Clean up
            a.dispose();
            b.dispose();
            result.dispose();
            
            console.log(`Matrix multiplication ${size}x${size}: ${duration}ms`);
        }
        
        // Check if we're using the optimal backend
        console.log('\n🎯 Backend Optimization Check:');
        console.log('='.repeat(50));
        
        const currentBackend = tf.getBackend();
        
        if (currentBackend === 'cpu') {
            console.log('✅ Using CPU backend (good for Node.js)');
            
            // Check if TensorFlow Node.js is available
            try {
                require('@tensorflow/tfjs-node');
                console.log('✅ @tensorflow/tfjs-node is installed (CPU optimized)');
            } catch (e) {
                console.log('⚠️ @tensorflow/tfjs-node not found - using browser backend');
                console.log('💡 Install with: npm install @tensorflow/tfjs-node');
            }
            
        } else if (currentBackend === 'tensorflow') {
            console.log('🚀 Using TensorFlow Node.js backend (OPTIMAL for CPU)');
        } else {
            console.log(`⚠️ Using ${currentBackend} backend (may not be optimal for Node.js)`);
        }
        
        // Check for GPU
        console.log('\n🎮 GPU Check:');
        console.log('='.repeat(50));
        
        try {
            require('@tensorflow/tfjs-node-gpu');
            console.log('🎮 @tensorflow/tfjs-node-gpu is installed');
            
            if (currentBackend === 'tensorflow') {
                console.log('ℹ️ GPU backend available but using CPU (this is fine for most cases)');
            }
        } catch (e) {
            console.log('💻 GPU backend not installed (CPU only)');
        }
        
        // Environment variables check
        console.log('\n🔧 Environment Variables:');
        console.log('='.repeat(50));
        
        const tfVars = [
            'TF_CPP_MIN_LOG_LEVEL',
            'TF_FORCE_GPU_ALLOW_GROWTH', 
            'TF_ENABLE_ONEDNN_OPTS',
            'NODE_OPTIONS'
        ];
        
        tfVars.forEach(varName => {
            const value = process.env[varName];
            if (value) {
                console.log(`${varName}: ${value}`);
            } else {
                console.log(`${varName}: not set`);
            }
        });
        
        // Recommendations
        console.log('\n💡 Recommendations:');
        console.log('='.repeat(50));
        
        if (currentBackend !== 'tensorflow') {
            console.log('⚠️ NOT using fastest CPU backend!');
            console.log('📝 To fix:');
            console.log('   1. Install: npm install @tensorflow/tfjs-node');
            console.log('   2. Import in your models: require("@tensorflow/tfjs-node");');
            console.log('   3. Set backend: await tf.setBackend("tensorflow");');
        } else {
            console.log('✅ Using optimal TensorFlow Node.js backend!');
        }
        
        // Check if OneDNN optimizations are enabled
        if (!process.env.TF_ENABLE_ONEDNN_OPTS) {
            console.log('💡 Enable OneDNN optimizations:');
            console.log('   Add to .env: TF_ENABLE_ONEDNN_OPTS=1');
        }
        
        if (!process.env.TF_CPP_MIN_LOG_LEVEL) {
            console.log('💡 Reduce TensorFlow logging:');
            console.log('   Add to .env: TF_CPP_MIN_LOG_LEVEL=2');
        }
        
    } catch (error) {
        console.error('❌ Error checking TensorFlow backend:', error.message);
    }
}

// Check what's actually imported in the models
function checkModelImports() {
    console.log('\n📚 Checking Model Imports:');
    console.log('='.repeat(50));
    
    const modelFiles = [
        'src/models/LSTMModel.js',
        'src/models/GRUModel.js', 
        'src/models/CNNModel.js',
        'src/models/TransformerModel.js'
    ];
    
    const fs = require('fs');
    const path = require('path');
    
    modelFiles.forEach(file => {
        const fullPath = path.join(process.cwd(), file);
        if (fs.existsSync(fullPath)) {
            const content = fs.readFileSync(fullPath, 'utf8');
            
            console.log(`\n📄 ${file}:`);
            
            // Check TensorFlow imports
            const tfImports = [
                "@tensorflow/tfjs-node",
                "@tensorflow/tfjs-backend-cpu", 
                "@tensorflow/tfjs"
            ];
            
            tfImports.forEach(imp => {
                if (content.includes(imp)) {
                    console.log(`  ✅ Imports: ${imp}`);
                } else {
                    console.log(`  ❌ Missing: ${imp}`);
                }
            });
            
        } else {
            console.log(`❌ File not found: ${file}`);
        }
    });
}

// Main execution
async function main() {
    await checkTensorFlowBackend();
    checkModelImports();
    
    console.log('\n🎯 Summary:');
    console.log('='.repeat(50));
    console.log('For fastest CPU performance, ensure:');
    console.log('1. @tensorflow/tfjs-node is installed');
    console.log('2. Models import: require("@tensorflow/tfjs-node");');  
    console.log('3. Backend is "tensorflow"');
    console.log('4. Environment variables are set for optimization');
}

if (require.main === module) {
    main().catch(console.error);
}

module.exports = { checkTensorFlowBackend };