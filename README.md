# EDSSP Remy-Go Bit Relay Ex-Machina Integration

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Go Version](https://img.shields.io/badge/Go-1.21+-blue.svg)](https://golang.org/)
[![Build Status](https://img.shields.io/badge/Build-Passing-green.svg)]()

An intelligent network optimization system that combines MIT's TCP ex Machina (Remy) machine learning algorithms with Williams-inspired fractal routing optimization and enterprise-grade security features.

## Overview

The EDSSP Remy-Go integration provides an advanced bit relay system that uses machine learning to optimize network performance in real-time. It combines several cutting-edge technologies:

- **Remy ML Agent**: Reinforcement learning for adaptive network optimization
- **Williams Fractal Routing**: Tree-based routing optimization with density field analysis
- **Guardian Security**: Real-time threat detection and response
- **BLAKE3 Verification**: Cryptographic integrity checking
- **Prometheus Metrics**: Comprehensive monitoring and observability

## Features

### 🧠 Machine Learning Optimization
- **Reinforcement Learning**: Adaptive network optimization using experience replay
- **Real-time Decision Making**: 10Hz learning loop for rapid adaptation
- **Multi-objective Optimization**: Balances RTT, throughput, and packet loss
- **Exploration vs Exploitation**: Configurable exploration rates for optimal learning

### 🌳 Williams Fractal Routing
- **Tree-based Evaluation**: Hierarchical routing decision making
- **Density Field Analysis**: Dynamic network topology optimization
- **Coherence Scoring**: Network stability measurement and optimization
- **Parity Distribution**: Intelligent data replication for fault tolerance

### 🔒 Enterprise Security
- **Guardian Integration**: Real-time security monitoring
- **BLAKE3 Hashing**: High-performance cryptographic verification
- **Anomaly Detection**: ML-based threat identification
- **Security Scoring**: Continuous risk assessment

### 📊 Observability
- **Prometheus Metrics**: Comprehensive performance monitoring
- **OpenTelemetry Tracing**: Distributed tracing support
- **Real-time Dashboards**: Performance visualization
- **Alerting Integration**: Automated incident response

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EDSSP Remy Agent                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────│
│  │   Remy ML   │  │  Williams   │  │  Guardian   │  │  Bit    │
│  │   Engine    │  │ Optimizer   │  │  Security   │  │ Relay   │
│  │             │  │             │  │             │  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────│
├─────────────────────────────────────────────────────────────────┤
│                    Network State Management                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────│
│  │   Routing   │  │ Performance │  │   Security  │  │  Parity │
│  │    Table    │  │  Metrics    │  │   Alerts    │  │  Queues │
│  │             │  │             │  │             │  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────│
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Go 1.21 or higher
- Access to the [Remy-Go repository](https://github.com/Aanthord/remy-go)
- Prometheus (for metrics collection)
- OpenTelemetry collector (optional, for tracing)

### Dependencies
```bash
go mod tidy
```

### Building
```bash
go build -o edssp-remy-agent .
```

## Configuration

### Basic Configuration
```json
{
  "learning_rate": 0.001,
  "exploration_rate": 0.1,
  "discount_factor": 0.99,
  "buffer_size": 10000,
  "network_topology": "mesh",
  "tree_depth": 8,
  "block_size": 256,
  "rtt_weight": 0.3,
  "similarity_weight": 0.4,
  "centrality_weight": 0.3,
  "min_replicas": 3,
  "max_replicas": 5,
  "replication_factor": 2,
  "max_queue_size": 1000,
  "initial_window_size": 10,
  "max_window_size": 1000,
  "ack_timeout_ms": 100,
  "blake3_verification": true,
  "guardian_integration": true,
  "threat_threshold": 0.5,
  "metrics_interval_ms": 1000,
  "log_level": "info",
  "enable_tracing": true
}
```

### Environment Variables
```bash
export EDSSP_CONFIG_PATH="/path/to/config.json"
export EDSSP_LOG_LEVEL="info"
export EDSSP_METRICS_PORT="8080"
export EDSSP_TRACING_ENDPOINT="http://localhost:14268/api/traces"
```

## Usage

### Basic Usage
```go
package main

import (
    "log"
    "os"
    "os/signal"
    "syscall"
)

func main() {
    config := &EDSSPRemyConfig{
        LearningRate:      0.001,
        ExplorationRate:   0.1,
        DiscountFactor:    0.99,
        BufferSize:        10000,
        NetworkTopology:   "mesh",
        TreeDepth:         8,
        BlockSize:         256,
        RTTWeight:         0.3,
        SimilarityWeight:  0.4,
        CentralityWeight:  0.3,
        MinReplicas:       3,
        MaxReplicas:       5,
        ReplicationFactor: 2,
        Blake3Verification: true,
        GuardianIntegration: true,
        ThreatThreshold:    0.5,
        EnableTracing:      true,
    }
    
    agent, err := NewEDSSPRemyAgent(config)
    if err != nil {
        log.Fatalf("Failed to create EDSSP Remy agent: %v", err)
    }
    
    if err := agent.Start(); err != nil {
        log.Fatalf("Failed to start EDSSP Remy agent: %v", err)
    }
    
    defer agent.Stop()
    
    // Wait for interrupt signal
    c := make(chan os.Signal, 1)
    signal.Notify(c, os.Interrupt, syscall.SIGTERM)
    <-c
    
    log.Println("Shutting down gracefully...")
}
```

### Advanced Configuration
```go
// Custom Williams optimization parameters
config.TreeDepth = 12
config.RTTWeight = 0.4
config.SimilarityWeight = 0.3
config.CentralityWeight = 0.3

// Enhanced security settings
config.Blake3Verification = true
config.GuardianIntegration = true
config.ThreatThreshold = 0.3 // More sensitive

// Performance tuning
config.BufferSize = 50000
config.MaxQueueSize = 5000
config.MaxWindowSize = 2000
```

## Monitoring and Metrics

### Prometheus Metrics
The agent exposes the following metrics:

**Learning Metrics:**
- `edssp_remy_learning_rate` - Current learning rate
- `edssp_remy_exploration_rate` - Current exploration rate
- `edssp_remy_training_loss` - Training loss
- `edssp_remy_episode_reward` - Reward per episode

**Performance Metrics:**
- `edssp_remy_rtt_optimization_ms` - RTT optimization achieved
- `edssp_remy_throughput_gain_percent` - Throughput improvement
- `edssp_remy_tree_eval_latency` - Williams optimization latency

**Security Metrics:**
- `edssp_remy_security_score` - Current security score
- `edssp_remy_guardian_alerts_total` - Total security alerts
- `edssp_remy_blake3_verifications_total` - Hash verifications

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "EDSSP Remy Agent Dashboard",
    "panels": [
      {
        "title": "Learning Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "edssp_remy_episode_reward",
            "legendFormat": "Episode Reward"
          }
        ]
      },
      {
        "title": "Network Optimization",
        "type": "graph",
        "targets": [
          {
            "expr": "edssp_remy_rtt_optimization_ms",
            "legendFormat": "RTT Optimization"
          }
        ]
      }
    ]
  }
}
```

## API Reference

### Core Agent Methods
```go
// Create new agent
func NewEDSSPRemyAgent(config *EDSSPRemyConfig) (*EDSSPRemyAgent, error)

// Start the agent
func (agent *EDSSPRemyAgent) Start() error

// Stop the agent gracefully
func (agent *EDSSPRemyAgent) Stop() error

// Get current network state
func (agent *EDSSPRemyAgent) GetNetworkState() *NetworkState

// Update configuration
func (agent *EDSSPRemyAgent) UpdateConfig(config *EDSSPRemyConfig) error
```

### Configuration Structures
```go
type EDSSPRemyConfig struct {
    LearningRate      float64 `json:"learning_rate"`
    ExplorationRate   float64 `json:"exploration_rate"`
    DiscountFactor    float64 `json:"discount_factor"`
    BufferSize        int     `json:"buffer_size"`
    NetworkTopology   string  `json:"network_topology"`
    TreeDepth         int     `json:"tree_depth"`
    BlockSize         int     `json:"block_size"`
    RTTWeight         float64 `json:"rtt_weight"`
    SimilarityWeight  float64 `json:"similarity_weight"`
    CentralityWeight  float64 `json:"centrality_weight"`
    MinReplicas       int     `json:"min_replicas"`
    MaxReplicas       int     `json:"max_replicas"`
    ReplicationFactor int     `json:"replication_factor"`
}
```

## Performance Tuning

### Learning Parameters
- **Learning Rate**: Start with 0.001, increase for faster adaptation
- **Exploration Rate**: Balance between exploration (0.1-0.3) and exploitation
- **Buffer Size**: Larger buffers (10k-50k) improve learning stability
- **Discount Factor**: 0.99 for long-term optimization, 0.9 for short-term

### Williams Optimization
- **Tree Depth**: 8-12 for most networks, higher for complex topologies
- **Weight Distribution**: Adjust RTT/Similarity/Centrality weights based on priorities
- **Block Size**: 256-1024 depending on network packet sizes

### Security Settings
- **Threat Threshold**: Lower values (0.3-0.5) for higher security
- **BLAKE3 Verification**: Enable for production environments
- **Guardian Integration**: Essential for enterprise deployments

## Troubleshooting

### Common Issues

**High CPU Usage:**
```bash
# Reduce learning frequency
config.LearningRate = 0.0005
config.MetricsInterval = 2000  # Reduce to 2 seconds
```

**Memory Leaks:**
```bash
# Reduce buffer sizes
config.BufferSize = 5000
config.MaxQueueSize = 500
```

**Security Alerts:**
```bash
# Check Guardian logs
grep "SECURITY" /var/log/edssp-remy.log

# Adjust threat threshold
config.ThreatThreshold = 0.7  # Less sensitive
```

### Debug Mode
```go
config.LogLevel = "debug"
config.EnableTracing = true
```

### Performance Profiling
```bash
go tool pprof http://localhost:8080/debug/pprof/profile
```

## License

This project is dual-licensed:

1. **AGPL-3.0** for research and academic use
2. **Commercial License** available for enterprise deployments

For commercial licensing, contact: michael.doran.808@gmail.com

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/your-repo/edssp-remy-go.git
cd edssp-remy-go
go mod tidy
go test ./...
```

## Acknowledgments

- **MIT CSAIL** for the original TCP ex Machina (Remy) research
- **[Remy-Go](https://github.com/Aanthord/remy-go)** project for the Go implementation
- **BLAKE3** team for the cryptographic hash function
- **Prometheus** team for the metrics system

## Support

- **Documentation**: [Link to full documentation]
- **Issues**: [GitHub Issues](https://github.com/your-repo/edssp-remy-go/issues)
- **Commercial Support**: michael.doran.808@gmail.com
- **Community**: [Discord/Slack channel]

## Roadmap

- [ ] Kubernetes operator for easy deployment
- [ ] Advanced ML models (transformer-based optimization)
- [ ] Multi-cloud support
- [ ] Enhanced security features
- [ ] Real-time configuration updates
- [ ] Advanced visualization dashboard

---

**EDSSP™ - Enterprise Distributed Systems Security Platform**  
Copyright (C) 2025 michael.doran.808@gmail.com
