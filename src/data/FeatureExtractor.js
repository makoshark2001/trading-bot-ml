const { Logger } = require('../utils');

class FeatureExtractor {
    constructor(config = {}) {
        this.indicators = config.indicators || [
            'rsi', 'macd', 'bollinger', 'ma', 'volume', 
            'stochastic', 'williamsR', 'ichimoku', 'adx', 'cci', 'parabolicSAR'
        ];
        this.lookbackPeriods = config.lookbackPeriods || [5, 10, 20];
        this.targetPeriods = config.targetPeriods || [1, 3, 5];
        
        Logger.info('FeatureExtractor initialized', {
            indicators: this.indicators.length,
            lookbackPeriods: this.lookbackPeriods,
            targetPeriods: this.targetPeriods
        });
    }
    
    extractFeatures(pairData) {
        try {
            const { history, strategies } = pairData;
            
            if (!history || !history.closes || history.closes.length < 60) {
                throw new Error('Insufficient historical data for feature extraction');
            }
            
            Logger.debug('Extracting features', {
                pair: pairData.pair,
                dataPoints: history.closes.length,
                strategiesAvailable: Object.keys(strategies || {}).length
            });
            
            const features = {};
            
            // 1. Price-based features
            features.price = this.extractPriceFeatures(history);
            
            // 2. Technical indicator features
            features.indicators = this.extractIndicatorFeatures(strategies);
            
            // 3. Volume features
            features.volume = this.extractVolumeFeatures(history);
            
            // 4. Volatility features
            features.volatility = this.extractVolatilityFeatures(history);
            
            // 5. Time-based features
            features.time = this.extractTimeFeatures(history);
            
            // Flatten all features into a single array
            const flatFeatures = this.flattenFeatures(features);
            
            Logger.debug('Feature extraction completed', {
                totalFeatures: flatFeatures.length,
                featureGroups: Object.keys(features)
            });
            
            return {
                features: flatFeatures,
                featureNames: this.getFeatureNames(features),
                metadata: {
                    pair: pairData.pair,
                    dataPoints: history.closes.length,
                    extractedAt: new Date().toISOString(),
                    featureCount: flatFeatures.length
                }
            };
            
        } catch (error) {
            Logger.error('Feature extraction failed', {
                error: error.message,
                pair: pairData.pair
            });
            throw error;
        }
    }
    
    extractPriceFeatures(history) {
        const closes = history.closes;
        const highs = history.highs;
        const lows = history.lows;
        const features = {};
        
        // Current price normalized features
        const currentPrice = closes[closes.length - 1];
        const priceRange = Math.max(...closes.slice(-20)) - Math.min(...closes.slice(-20));
        
        features.currentPrice = currentPrice;
        features.pricePosition = priceRange > 0 ? (currentPrice - Math.min(...closes.slice(-20))) / priceRange : 0.5;
        
        // Returns for different periods
        this.lookbackPeriods.forEach(period => {
            if (closes.length > period) {
                const prevPrice = closes[closes.length - period - 1];
                features[`return_${period}`] = prevPrice > 0 ? (currentPrice - prevPrice) / prevPrice : 0;
            }
        });
        
        // High-Low spread
        features.hlSpread = highs[highs.length - 1] - lows[lows.length - 1];
        features.hlPosition = (currentPrice - lows[lows.length - 1]) / (highs[highs.length - 1] - lows[lows.length - 1]);
        
        return features;
    }
    
    extractIndicatorFeatures(strategies) {
        const features = {};
        
        this.indicators.forEach(indicator => {
            if (strategies[indicator] && !strategies[indicator].error) {
                const strategy = strategies[indicator];
                
                // Common features for all indicators
                features[`${indicator}_confidence`] = strategy.confidence || 0;
                features[`${indicator}_strength`] = strategy.strength || 0;
                
                // Suggestion encoding (buy=1, sell=-1, hold=0)
                features[`${indicator}_suggestion`] = 
                    strategy.suggestion === 'buy' ? 1 : 
                    strategy.suggestion === 'sell' ? -1 : 0;
                
                // Indicator-specific features
                switch(indicator) {
                    case 'rsi':
                        features.rsi_value = strategy.value || 50;
                        features.rsi_overbought = strategy.value > 70 ? 1 : 0;
                        features.rsi_oversold = strategy.value < 30 ? 1 : 0;
                        break;
                        
                    case 'macd':
                        features.macd_line = strategy.macdLine || 0;
                        features.macd_signal = strategy.signalLine || 0;
                        features.macd_histogram = strategy.histogram || 0;
                        break;
                        
                    case 'bollinger':
                        features.bb_percentB = strategy.percentB || 0.5;
                        features.bb_bandwidth = strategy.bandwidth || 0;
                        features.bb_position = strategy.percentB > 0.8 ? 1 : strategy.percentB < 0.2 ? -1 : 0;
                        break;
                        
                    case 'ma':
                        features.ma_spread = strategy.spreadPercent || 0;
                        features.ma_trend = strategy.metadata?.trend === 'uptrend' ? 1 : 
                                          strategy.metadata?.trend === 'downtrend' ? -1 : 0;
                        break;
                        
                    case 'volume':
                        features.volume_ratio = strategy.volumeRatio || 1;
                        features.volume_spike = strategy.metadata?.volumeSpike ? 1 : 0;
                        break;
                        
                    case 'stochastic':
                        features.stoch_k = strategy.k || 50;
                        features.stoch_d = strategy.d || 50;
                        features.stoch_overbought = strategy.k > 80 ? 1 : 0;
                        features.stoch_oversold = strategy.k < 20 ? 1 : 0;
                        break;
                        
                    case 'williamsR':
                        features.williams_value = strategy.value || -50;
                        features.williams_level = strategy.metadata?.level === 'Overbought' ? 1 : 
                                                strategy.metadata?.level === 'Oversold' ? -1 : 0;
                        break;
                        
                    case 'ichimoku':
                        features.ichimoku_cloud = strategy.cloudColor === 'bullish' ? 1 : 
                                                strategy.cloudColor === 'bearish' ? -1 : 0;
                        features.ichimoku_thickness = strategy.cloudThickness || 0;
                        features.ichimoku_trend = strategy.metadata?.trend === 'strong_uptrend' ? 2 :
                                                strategy.metadata?.trend === 'uptrend' ? 1 :
                                                strategy.metadata?.trend === 'strong_downtrend' ? -2 :
                                                strategy.metadata?.trend === 'downtrend' ? -1 : 0;
                        break;
                        
                    case 'adx':
                        features.adx_value = strategy.adx || 0;
                        features.adx_plus_di = strategy.plusDI || 0;
                        features.adx_minus_di = strategy.minusDI || 0;
                        features.adx_trend_strength = strategy.adx > 25 ? 1 : 0;
                        break;
                        
                    case 'cci':
                        features.cci_value = strategy.cci || 0;
                        features.cci_overbought = strategy.cci > 100 ? 1 : 0;
                        features.cci_oversold = strategy.cci < -100 ? 1 : 0;
                        break;
                        
                    case 'parabolicSAR':
                        features.sar_trend = strategy.trend === 'uptrend' ? 1 : -1;
                        features.sar_reversal = strategy.metadata?.reversal ? 1 : 0;
                        features.sar_af = strategy.af || 0;
                        break;
                }
            }
        });
        
        return features;
    }
    
    extractVolumeFeatures(history) {
        const volumes = history.volumes;
        const features = {};
        
        if (volumes && volumes.length > 0) {
            const currentVolume = volumes[volumes.length - 1];
            const avgVolume = volumes.slice(-20).reduce((sum, v) => sum + v, 0) / Math.min(20, volumes.length);
            
            features.current_volume = currentVolume;
            features.volume_ma_ratio = avgVolume > 0 ? currentVolume / avgVolume : 1;
            
            // Volume trend
            if (volumes.length >= 5) {
                const recent = volumes.slice(-5);
                const older = volumes.slice(-10, -5);
                const recentAvg = recent.reduce((sum, v) => sum + v, 0) / recent.length;
                const olderAvg = older.reduce((sum, v) => sum + v, 0) / older.length;
                features.volume_trend = olderAvg > 0 ? (recentAvg - olderAvg) / olderAvg : 0;
            }
        }
        
        return features;
    }
    
    extractVolatilityFeatures(history) {
        const closes = history.closes;
        const features = {};
        
        // Calculate returns
        const returns = [];
        for (let i = 1; i < closes.length; i++) {
            if (closes[i - 1] > 0) {
                returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
            }
        }
        
        if (returns.length > 0) {
            // Standard deviation (volatility)
            const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
            const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
            features.volatility = Math.sqrt(variance);
            
            // Recent volatility vs historical
            const recentReturns = returns.slice(-10);
            const recentMean = recentReturns.reduce((sum, r) => sum + r, 0) / recentReturns.length;
            const recentVariance = recentReturns.reduce((sum, r) => sum + Math.pow(r - recentMean, 2), 0) / recentReturns.length;
            const recentVolatility = Math.sqrt(recentVariance);
            
            features.recent_volatility = recentVolatility;
            features.volatility_ratio = features.volatility > 0 ? recentVolatility / features.volatility : 1;
        }
        
        return features;
    }
    
    extractTimeFeatures(history) {
        const features = {};
        
        if (history.timestamps && history.timestamps.length > 0) {
            const currentTime = new Date(history.timestamps[history.timestamps.length - 1]);
            
            // Time of day (0-23 normalized to 0-1)
            features.hour_of_day = currentTime.getHours() / 23;
            
            // Day of week (0-6 normalized to 0-1)
            features.day_of_week = currentTime.getDay() / 6;
            
            // Sine/cosine encoding for cyclical features
            features.hour_sin = Math.sin(2 * Math.PI * currentTime.getHours() / 24);
            features.hour_cos = Math.cos(2 * Math.PI * currentTime.getHours() / 24);
            features.day_sin = Math.sin(2 * Math.PI * currentTime.getDay() / 7);
            features.day_cos = Math.cos(2 * Math.PI * currentTime.getDay() / 7);
        }
        
        return features;
    }
    
    flattenFeatures(features) {
        const flattened = [];
        
        Object.values(features).forEach(group => {
            Object.values(group).forEach(value => {
                // Handle potential null/undefined values
                const numericValue = typeof value === 'number' && isFinite(value) ? value : 0;
                flattened.push(numericValue);
            });
        });
        
        return flattened;
    }
    
    getFeatureNames(features) {
        const names = [];
        
        Object.entries(features).forEach(([groupName, group]) => {
            Object.keys(group).forEach(featureName => {
                names.push(`${groupName}_${featureName}`);
            });
        });
        
        return names;
    }
    
    // Create training targets for price prediction
    createTargets(history, targetPeriods = [1, 3, 5]) {
        const closes = history.closes;
        const targets = {};
        
        targetPeriods.forEach(period => {
            targets[`price_change_${period}`] = [];
            targets[`direction_${period}`] = [];
            
            for (let i = 0; i < closes.length - period; i++) {
                const currentPrice = closes[i];
                const futurePrice = closes[i + period];
                
                if (currentPrice > 0) {
                    const priceChange = (futurePrice - currentPrice) / currentPrice;
                    targets[`price_change_${period}`].push(priceChange);
                    targets[`direction_${period}`].push(priceChange > 0 ? 1 : 0);
                }
            }
        });
        
        return targets;
    }
}

module.exports = FeatureExtractor;