const axios = require('axios');
const config = require('config');
const { Logger } = require('../utils');

class DataClient {
    constructor() {
        this.baseUrl = config.get('core.baseUrl');
        this.endpoints = config.get('core.endpoints');
        
        // Create axios instance with enhanced timeout configuration
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: 60000, // Increased to 60 seconds
            headers: {
                'Content-Type': 'application/json',
                'Connection': 'keep-alive',
                'Accept': 'application/json'
            },
            // Enhanced retry configuration
            maxRedirects: 5,
            validateStatus: function (status) {
                return status >= 200 && status < 300;
            },
            // Add keep-alive agent for better connection handling
            httpAgent: require('http').Agent({ 
                keepAlive: true,
                timeout: 60000,
                keepAliveMsecs: 1000
            }),
            httpsAgent: require('https').Agent({ 
                keepAlive: true,
                timeout: 60000,
                keepAliveMsecs: 1000
            })
        });
        
        // Add request interceptor for logging
        this.client.interceptors.request.use(
            (config) => {
                Logger.debug('Making request to core service', {
                    url: config.url,
                    method: config.method,
                    baseURL: config.baseURL,
                    timeout: config.timeout
                });
                return config;
            },
            (error) => {
                Logger.error('Request interceptor error', { error: error.message });
                return Promise.reject(error);
            }
        );
        
        // Add response interceptor for enhanced error handling
        this.client.interceptors.response.use(
            (response) => {
                Logger.debug('Received response from core service', {
                    status: response.status,
                    url: response.config.url,
                    dataSize: JSON.stringify(response.data).length,
                    responseTime: response.headers['x-response-time'] || 'unknown'
                });
                return response;
            },
            (error) => {
                const errorInfo = {
                    message: error.message,
                    code: error.code,
                    status: error.response?.status,
                    url: error.config?.url,
                    timeout: error.config?.timeout
                };
                
                if (error.code === 'ECONNRESET') {
                    errorInfo.type = 'CONNECTION_RESET';
                } else if (error.code === 'ECONNREFUSED') {
                    errorInfo.type = 'CONNECTION_REFUSED';
                } else if (error.code === 'ETIMEDOUT') {
                    errorInfo.type = 'TIMEOUT';
                } else if (error.message.includes('timeout')) {
                    errorInfo.type = 'TIMEOUT';
                }
                
                Logger.error('Response interceptor error', errorInfo);
                return Promise.reject(error);
            }
        );
        
        Logger.info('DataClient initialized with enhanced timeout handling', {
            baseUrl: this.baseUrl,
            timeout: 60000,
            keepAlive: true
        });
    }
    
    async getAllData() {
        try {
            Logger.debug('Fetching all data from core service');
            
            const response = await this.retryRequest(() => 
                this.client.get(this.endpoints.data, { timeout: 45000 })
            );
            
            if (!response.data) {
                throw new Error('No data received from core service');
            }
            
            Logger.debug('Successfully fetched all data', {
                pairs: response.data.pairs?.length || 0,
                status: response.data.status,
                dataSize: JSON.stringify(response.data).length
            });
            
            return response.data;
        } catch (error) {
            Logger.error('Failed to fetch all data from core', {
                error: error.message,
                code: error.code,
                type: this.getErrorType(error),
                url: this.baseUrl + this.endpoints.data
            });
            throw new Error(`Core service unavailable: ${error.message}`);
        }
    }
    
    async getPairData(pair) {
        try {
            Logger.debug(`Fetching data for pair: ${pair}`);
            
            const response = await this.retryRequest(() => 
                this.client.get(`${this.endpoints.pair}/${pair.toUpperCase()}`, { 
                    timeout: 30000 
                })
            );
            
            if (!response.data) {
                throw new Error(`No data received for pair ${pair}`);
            }
            
            Logger.debug(`Successfully fetched data for ${pair}`, {
                hasHistory: !!response.data.history,
                hasStrategies: !!response.data.strategies,
                dataPoints: response.data.history?.closes?.length || 0
            });
            
            // Validate response data
            if (!response.data.history || !response.data.history.closes) {
                throw new Error(`Invalid data structure for pair ${pair}`);
            }
            
            if (response.data.history.closes.length < 60) {
                throw new Error(`Insufficient data for pair ${pair}: ${response.data.history.closes.length} points`);
            }
            
            return response.data;
        } catch (error) {
            Logger.error(`Failed to fetch data for pair ${pair}`, {
                error: error.message,
                code: error.code,
                type: this.getErrorType(error),
                url: this.baseUrl + this.endpoints.pair + '/' + pair
            });
            throw new Error(`Failed to get data for ${pair}: ${error.message}`);
        }
    }
    
    async checkCoreHealth() {
        try {
            Logger.debug('Checking core service health');
            
            const response = await this.retryRequest(() => 
                this.client.get(this.endpoints.health, { timeout: 15000 })
            );
            
            if (!response.data) {
                throw new Error('No health data received');
            }
            
            const isHealthy = response.data.status === 'healthy';
            Logger.info('Core service health check', {
                status: response.data.status,
                healthy: isHealthy,
                dataCollection: response.data.dataCollection,
                responseTime: response.headers['x-response-time'] || 'unknown'
            });
            
            return response.data;
        } catch (error) {
            Logger.error('Core service health check failed', {
                error: error.message,
                code: error.code,
                type: this.getErrorType(error),
                url: this.baseUrl + this.endpoints.health
            });
            throw new Error(`Core service health check failed: ${error.message}`);
        }
    }
    
    async waitForCoreService(maxRetries = 20, initialDelay = 2000) {
        Logger.info('Waiting for core service to become available...', {
            maxRetries,
            initialDelay,
            coreUrl: this.baseUrl
        });
        
        for (let i = 0; i < maxRetries; i++) {
            try {
                await this.checkCoreHealth();
                Logger.info('Core service is ready', {
                    attempt: i + 1,
                    maxRetries
                });
                return true;
            } catch (error) {
                const delay = initialDelay * Math.min(2, (i + 1)); // Progressive delay
                
                Logger.warn(`Core service not ready, attempt ${i + 1}/${maxRetries}`, {
                    error: error.message,
                    code: error.code,
                    type: this.getErrorType(error),
                    retryIn: delay / 1000 + 's',
                    nextAttempt: i < maxRetries - 1 ? 'yes' : 'no'
                });
                
                if (i < maxRetries - 1) {
                    await this.sleep(delay);
                }
            }
        }
        
        throw new Error(`Core service failed to become ready within ${maxRetries} attempts`);
    }
    
    // Enhanced retry mechanism with exponential backoff
    async retryRequest(requestFunction, maxRetries = 5, baseDelay = 2000) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                const result = await requestFunction();
                
                // If we get here, the request succeeded
                if (attempt > 1) {
                    Logger.info(`Request succeeded on attempt ${attempt}/${maxRetries}`);
                }
                
                return result;
                
            } catch (error) {
                lastError = error;
                const errorType = this.getErrorType(error);
                
                Logger.warn(`Request attempt ${attempt}/${maxRetries} failed`, {
                    error: error.message,
                    code: error.code,
                    status: error.response?.status,
                    type: errorType,
                    willRetry: attempt < maxRetries && this.shouldRetry(error)
                });
                
                // Don't retry on certain error types
                if (!this.shouldRetry(error)) {
                    Logger.info('Not retrying due to error type', { 
                        code: error.code,
                        status: error.response?.status,
                        type: errorType
                    });
                    throw error;
                }
                
                // Don't wait after the last attempt
                if (attempt < maxRetries) {
                    const delay = baseDelay * Math.pow(1.5, attempt - 1); // Exponential backoff
                    Logger.info(`Retrying in ${delay}ms... (attempt ${attempt + 1}/${maxRetries})`);
                    await this.sleep(delay);
                }
            }
        }
        
        // All retries failed
        Logger.error(`All ${maxRetries} attempts failed`, { 
            lastError: lastError.message,
            code: lastError.code,
            type: this.getErrorType(lastError)
        });
        throw lastError;
    }
    
    // Determine error type for better handling
    getErrorType(error) {
        if (!error) return 'UNKNOWN';
        
        if (error.code === 'ECONNRESET') return 'CONNECTION_RESET';
        if (error.code === 'ECONNREFUSED') return 'CONNECTION_REFUSED';
        if (error.code === 'ETIMEDOUT') return 'TIMEOUT';
        if (error.code === 'ENOTFOUND') return 'DNS_ERROR';
        if (error.code === 'ECONNABORTED') return 'CONNECTION_ABORTED';
        if (error.message?.includes('timeout')) return 'TIMEOUT';
        if (error.message?.includes('ECONNRESET')) return 'CONNECTION_RESET';
        if (error.response?.status >= 500) return 'SERVER_ERROR';
        if (error.response?.status >= 400) return 'CLIENT_ERROR';
        
        return 'NETWORK_ERROR';
    }
    
    // Determine if we should retry based on error type
    shouldRetry(error) {
        const errorType = this.getErrorType(error);
        
        // Always retry on network and timeout errors
        if (['CONNECTION_RESET', 'TIMEOUT', 'NETWORK_ERROR', 'CONNECTION_REFUSED'].includes(errorType)) {
            return true;
        }
        
        // Retry on server errors (5xx)
        if (error.response?.status >= 500) {
            return true;
        }
        
        // Retry on specific client errors
        if ([408, 429, 502, 503, 504].includes(error.response?.status)) {
            return true;
        }
        
        // Don't retry on other client errors (4xx) or successful responses
        return false;
    }
    
    // Helper function for delays
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // Test connection to core service with comprehensive diagnostics
    async testConnection() {
        try {
            Logger.info('Testing connection to core service...', {
                baseUrl: this.baseUrl,
                timeout: this.client.defaults.timeout
            });
            
            const startTime = Date.now();
            const health = await this.checkCoreHealth();
            const responseTime = Date.now() - startTime;
            
            Logger.info('Connection test successful', {
                responseTime: `${responseTime}ms`,
                coreStatus: health.status,
                dataPoints: health.dataCollection?.totalDataPoints || 0,
                isCollecting: health.dataCollection?.isCollecting || false
            });
            
            return {
                success: true,
                responseTime,
                status: health.status,
                dataCollection: health.dataCollection,
                timestamp: Date.now()
            };
            
        } catch (error) {
            Logger.error('Connection test failed', { 
                error: error.message,
                code: error.code,
                type: this.getErrorType(error)
            });
            return {
                success: false,
                error: error.message,
                code: error.code,
                type: this.getErrorType(error),
                timestamp: Date.now()
            };
        }
    }
    
    // Get available pairs from core service
    async getAvailablePairs() {
        try {
            Logger.debug('Fetching available pairs from core service');
            
            const response = await this.retryRequest(() => 
                this.client.get('/api/pairs', { timeout: 20000 })
            );
            
            Logger.debug('Successfully fetched available pairs', {
                count: response.data.pairs?.length || 0
            });
            
            return response.data;
        } catch (error) {
            Logger.error('Failed to fetch available pairs', {
                error: error.message,
                code: error.code,
                type: this.getErrorType(error)
            });
            throw new Error(`Failed to get available pairs: ${error.message}`);
        }
    }
    
    // Get specific indicator data for a pair
    async getIndicatorData(pair, indicator) {
        try {
            Logger.debug(`Fetching ${indicator} data for pair: ${pair}`);
            
            const response = await this.retryRequest(() => 
                this.client.get(
                    `${this.endpoints.pair}/${pair.toUpperCase()}/indicator/${indicator}`,
                    { timeout: 15000 }
                )
            );
            
            Logger.debug(`Successfully fetched ${indicator} data for ${pair}`);
            
            return response.data;
        } catch (error) {
            Logger.error(`Failed to fetch ${indicator} data for pair ${pair}`, {
                error: error.message,
                code: error.code,
                type: this.getErrorType(error)
            });
            throw new Error(`Failed to get ${indicator} data for ${pair}: ${error.message}`);
        }
    }
    
    // Get core service configuration
    async getCoreConfig() {
        try {
            Logger.debug('Fetching core service configuration');
            
            const response = await this.retryRequest(() => 
                this.client.get('/api/config', { timeout: 10000 })
            );
            
            Logger.debug('Successfully fetched core configuration');
            
            return response.data;
        } catch (error) {
            Logger.error('Failed to fetch core configuration', {
                error: error.message,
                code: error.code,
                type: this.getErrorType(error)
            });
            throw new Error(`Failed to get core configuration: ${error.message}`);
        }
    }
    
    // Check if pair data is ready for ML processing
    async isPairReadyForML(pair) {
        try {
            const pairData = await this.getPairData(pair);
            
            const isReady = 
                pairData.history &&
                pairData.history.closes &&
                pairData.history.closes.length >= 60 &&
                pairData.strategies &&
                Object.keys(pairData.strategies).length > 0;
            
            Logger.debug(`ML readiness check for ${pair}`, {
                isReady,
                dataPoints: pairData.history?.closes?.length || 0,
                strategies: Object.keys(pairData.strategies || {}).length
            });
            
            return {
                ready: isReady,
                dataPoints: pairData.history?.closes?.length || 0,
                strategies: Object.keys(pairData.strategies || {}).length,
                pair: pair
            };
            
        } catch (error) {
            Logger.error(`ML readiness check failed for ${pair}`, { 
                error: error.message,
                code: error.code,
                type: this.getErrorType(error)
            });
            return {
                ready: false,
                error: error.message,
                pair: pair
            };
        }
    }
    
    // Get connection status and diagnostics
    getConnectionStatus() {
        return {
            baseUrl: this.baseUrl,
            endpoints: this.endpoints,
            timeout: this.client.defaults.timeout,
            keepAlive: true,
            retryConfig: {
                maxRetries: 5,
                baseDelay: 2000,
                exponentialBackoff: true
            },
            timestamp: Date.now()
        };
    }
    
    // Force connection reset (useful for debugging)
    resetConnection() {
        Logger.info('Resetting DataClient connection');
        
        // Recreate the axios instance
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: 60000,
            headers: {
                'Content-Type': 'application/json',
                'Connection': 'keep-alive',
                'Accept': 'application/json'
            },
            maxRedirects: 5,
            validateStatus: function (status) {
                return status >= 200 && status < 300;
            },
            httpAgent: require('http').Agent({ 
                keepAlive: true,
                timeout: 60000,
                keepAliveMsecs: 1000
            }),
            httpsAgent: require('https').Agent({ 
                keepAlive: true,
                timeout: 60000,
                keepAliveMsecs: 1000
            })
        });
        
        Logger.info('DataClient connection reset completed');
    }
}

module.exports = DataClient;