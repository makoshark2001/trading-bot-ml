const os = require('os');
const { execSync } = require('child_process');

async function diagnoseGPUSupport() {
    console.log('🔍 GPU Support Diagnostic Tool');
    console.log('=' * 50);
    
    // System Information
    console.log('\n💻 System Information:');
    console.log('-' * 25);
    console.log('Platform:', os.platform());
    console.log('Architecture:', os.arch());
    console.log('Node.js Version:', process.version);
    console.log('OS Release:', os.release());
    console.log('Total Memory:', Math.round(os.totalmem() / 1024 / 1024 / 1024) + 'GB');
    
    // Check installed packages
    console.log('\n📦 TensorFlow.js Packages:');
    console.log('-' * 30);
    
    const packages = [
        '@tensorflow/tfjs',
        '@tensorflow/tfjs-backend-cpu',
        '@tensorflow/tfjs-backend-webgl',
        '@tensorflow/tfjs-node',
        '@tensorflow/tfjs-node-gpu'
    ];
    
    for (const pkg of packages) {
        try {
            const version = require(`${pkg}/package.json`).version;
            console.log(`✅ ${pkg}: v${version}`);
        } catch (error) {
            console.log(`❌ ${pkg}: Not installed`);
        }
    }
    
    // GPU Hardware Detection
    console.log('\n🎮 GPU Hardware Detection:');
    console.log('-' * 32);
    
    await checkGPUHardware();
    
    // CUDA Detection (Windows/Linux)
    console.log('\n⚡ CUDA/GPU Software:');
    console.log('-' * 25);
    
    await checkCUDA();
    
    // TensorFlow Backend Testing
    console.log('\n🧪 TensorFlow Backend Testing:');
    console.log('-' * 35);
    
    await testTensorFlowBackends();
    
    // Recommendations
    console.log('\n💡 Recommendations:');
    console.log('-' * 20);
    
    await provideRecommendations();
}

async function checkGPUHardware() {
    try {
        if (os.platform() === 'win32') {
            // Windows GPU detection
            try {
                const output = execSync('wmic path win32_VideoController get name', { encoding: 'utf8' });
                const gpus = output.split('\n')
                    .filter(line => line.trim() && !line.includes('Name'))
                    .map(line => line.trim());
                
                if (gpus.length > 0) {
                    console.log('Windows GPU(s) detected:');
                    gpus.forEach((gpu, index) => {
                        console.log(`  ${index + 1}. ${gpu}`);
                        
                        // Check if it's NVIDIA (CUDA compatible)
                        if (gpu.toLowerCase().includes('nvidia')) {
                            console.log('     ✅ NVIDIA GPU (potentially CUDA compatible)');
                        } else if (gpu.toLowerCase().includes('amd') || gpu.toLowerCase().includes('radeon')) {
                            console.log('     ⚠️  AMD GPU (limited TensorFlow.js support)');
                        } else if (gpu.toLowerCase().includes('intel')) {
                            console.log('     ⚠️  Intel GPU (limited TensorFlow.js support)');
                        }
                    });
                } else {
                    console.log('❌ No GPUs detected via WMI');
                }
            } catch (error) {
                console.log('❌ Windows GPU detection failed:', error.message);
            }
        } else if (os.platform() === 'linux') {
            // Linux GPU detection
            try {
                const output = execSync('lspci | grep -i vga', { encoding: 'utf8' });
                console.log('Linux GPU(s) detected:');
                console.log(output.trim());
            } catch (error) {
                console.log('❌ Linux GPU detection failed. Try: sudo apt install pciutils');
            }
        } else if (os.platform() === 'darwin') {
            // macOS GPU detection
            try {
                const output = execSync('system_profiler SPDisplaysDataType', { encoding: 'utf8' });
                const gpuInfo = output.match(/Chipset Model: (.+)/g);
                if (gpuInfo) {
                    console.log('macOS GPU(s) detected:');
                    gpuInfo.forEach(gpu => console.log(`  ${gpu}`));
                } else {
                    console.log('❌ No GPU info found on macOS');
                }
            } catch (error) {
                console.log('❌ macOS GPU detection failed:', error.message);
            }
        }
    } catch (error) {
        console.log('❌ Hardware detection failed:', error.message);
    }
}

async function checkCUDA() {
    try {
        // Check NVIDIA-SMI (shows CUDA driver)
        try {
            const nvidiaSmi = execSync('nvidia-smi --version', { encoding: 'utf8' });
            console.log('✅ NVIDIA Driver detected:');
            console.log('  ', nvidiaSmi.split('\n')[0]);
        } catch (error) {
            console.log('❌ nvidia-smi not found (NVIDIA drivers not installed)');
        }
        
        // Check CUDA toolkit
        try {
            const cudaVersion = execSync('nvcc --version', { encoding: 'utf8' });
            const versionMatch = cudaVersion.match(/release (\d+\.\d+)/);
            if (versionMatch) {
                console.log('✅ CUDA Toolkit detected:', versionMatch[1]);
            } else {
                console.log('✅ CUDA Toolkit found but version unclear');
            }
        } catch (error) {
            console.log('❌ nvcc not found (CUDA toolkit not installed)');
        }
        
        // Check cuDNN (if possible)
        try {
            const ldConfigOutput = execSync('ldconfig -p | grep cudnn || echo "not found"', { encoding: 'utf8' });
            if (!ldConfigOutput.includes('not found')) {
                console.log('✅ cuDNN libraries detected');
            } else {
                console.log('❌ cuDNN libraries not found');
            }
        } catch (error) {
            console.log('ℹ️  cuDNN check not available on this platform');
        }
        
    } catch (error) {
        console.log('❌ CUDA detection failed:', error.message);
    }
}

async function testTensorFlowBackends() {
    try {
        const tf = require('@tensorflow/tfjs');
        
        // Test available backends
        console.log('Testing TensorFlow.js backends...\n');
        
        const backendsToTest = ['cpu', 'webgl', 'nodejs-gpu'];
        
        for (const backend of backendsToTest) {
            try {
                console.log(`Testing ${backend} backend:`);
                
                // Try to require the backend
                if (backend === 'webgl') {
                    require('@tensorflow/tfjs-backend-webgl');
                } else if (backend === 'nodejs-gpu') {
                    require('@tensorflow/tfjs-node-gpu');
                }
                
                // Try to set the backend
                await tf.setBackend(backend);
                await tf.ready();
                
                // Test with a simple operation
                const startTime = Date.now();
                const a = tf.ones([100, 100]);
                const b = tf.ones([100, 100]);
                const result = tf.matMul(a, b);
                await result.data();
                const duration = Date.now() - startTime;
                
                // Clean up
                a.dispose();
                b.dispose();
                result.dispose();
                
                console.log(`  ✅ ${backend}: Working (${duration}ms)`);
                console.log(`     Backend: ${tf.getBackend()}`);
                console.log(`     Memory: ${tf.memory().numBytes} bytes\n`);
                
            } catch (error) {
                console.log(`  ❌ ${backend}: Failed - ${error.message}\n`);
            }
        }
        
        // Reset to CPU
        await tf.setBackend('cpu');
        await tf.ready();
        
    } catch (error) {
        console.log('❌ TensorFlow backend testing failed:', error.message);
    }
}

async function provideRecommendations() {
    const platform = os.platform();
    
    console.log('Based on your system, here are recommendations:\n');
    
    // General recommendations
    console.log('🎯 For GPU acceleration with TensorFlow.js:');
    console.log('   1. You need an NVIDIA GPU with CUDA support');
    console.log('   2. Install NVIDIA drivers');
    console.log('   3. Install CUDA toolkit (11.x or 12.x)');
    console.log('   4. Install cuDNN (optional but recommended)');
    console.log('   5. Use @tensorflow/tfjs-node-gpu package\n');
    
    // Platform-specific recommendations
    if (platform === 'win32') {
        console.log('🪟 Windows-specific steps:');
        console.log('   1. Download NVIDIA drivers from nvidia.com');
        console.log('   2. Download CUDA toolkit from developer.nvidia.com');
        console.log('   3. Add CUDA to your PATH environment variable');
        console.log('   4. Restart your system');
        console.log('   5. Verify with: nvidia-smi and nvcc --version\n');
    } else if (platform === 'linux') {
        console.log('🐧 Linux-specific steps:');
        console.log('   1. sudo apt update && sudo apt install nvidia-driver-XXX');
        console.log('   2. Download and install CUDA toolkit');
        console.log('   3. Add CUDA to ~/.bashrc: export PATH=/usr/local/cuda/bin:$PATH');
        console.log('   4. sudo ldconfig');
        console.log('   5. Restart and verify: nvidia-smi\n');
    } else if (platform === 'darwin') {
        console.log('🍎 macOS Note:');
        console.log('   • CUDA is not supported on modern macOS');
        console.log('   • Consider using Metal Performance Shaders (not supported in tfjs-node)');
        console.log('   • CPU performance on Apple Silicon is quite good\n');
    }
    
    console.log('⚡ Alternative options:');
    console.log('   • Use cloud GPU instances (Google Colab, AWS, etc.)');
    console.log('   • Consider @tensorflow/tfjs-node (CPU-optimized)');
    console.log('   • CPU performance is often sufficient for small models');
    console.log('   • WebGL backend works in browsers with GPU support\n');
    
    console.log('🔧 If you have NVIDIA GPU but still not working:');
    console.log('   • Check CUDA version compatibility with TensorFlow.js');
    console.log('   • Try different versions of @tensorflow/tfjs-node-gpu');
    console.log('   • Verify GPU memory is not exhausted');
    console.log('   • Check Windows/Linux GPU drivers are up to date');
}

// System environment check
function checkEnvironmentVariables() {
    console.log('\n🌍 Environment Variables:');
    console.log('-' * 25);
    
    const gpuEnvVars = [
        'CUDA_VISIBLE_DEVICES',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'TF_CPP_MIN_LOG_LEVEL',
        'NVIDIA_VISIBLE_DEVICES'
    ];
    
    gpuEnvVars.forEach(envVar => {
        const value = process.env[envVar];
        if (value) {
            console.log(`✅ ${envVar}: ${value}`);
        } else {
            console.log(`❌ ${envVar}: Not set`);
        }
    });
}

async function main() {
    try {
        await diagnoseGPUSupport();
        checkEnvironmentVariables();
        
        console.log('\n🎉 Diagnostic completed!');
        console.log('\nNote: GPU acceleration is optional. Your CPU-based ML training');
        console.log('is working perfectly and may be sufficient for your needs.');
        
    } catch (error) {
        console.error('Diagnostic failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { diagnoseGPUSupport };