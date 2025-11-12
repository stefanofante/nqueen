# Changelog

All notable changes to the N-Queens Comparative Algorithm Analysis project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-11-12

### Changed

- Modularized the orchestration layer into `nqueens/analysis/` with dedicated modules:
  - `settings`, `stats`, `tuning`, `experiments`, `reporting`, `plots`, `cli`
- Converted `algoanalisys.py` into a thin backwards-compatible facade re-exporting the public APIs
- Made plotting imports optional with clear runtime stubs when `matplotlib` is not installed
- Updated README to document the new modular architecture, Python API usage, and optional plotting dependency

### Fixed

- Ensured unit tests and the quick regression runner work without `matplotlib` installed

## [2.0.0] - 2025-11-11

### Major Changes

- **BREAKING**: Complete internationalization - all code documentation, comments, and user interface converted to English
- **BREAKING**: Restructured output to professional scientific format

### Added

- Comprehensive English documentation with technical depth
- Professional README with detailed algorithm descriptions
- Advanced statistical analysis with 60+ high-quality charts
- Comprehensive visualization framework:
  - 9 base analysis charts per fitness function
  - 6 fitness function comparison charts  
  - 12+ statistical analysis charts with box plots and histograms
  - 10+ parameter tuning analysis charts with heatmaps
- Theoretical vs practical cost correlation analysis
- Algorithm stability analysis with variance metrics
- Failure quality assessment for unsuccessful runs
- Parameter sensitivity analysis for genetic algorithm
- Pareto efficiency analysis for fitness functions

### Enhanced

- Statistical engine with advanced descriptive statistics
- Parallel processing framework with intelligent resource management
- Timeout management system with detailed categorization
- CSV export system with comprehensive data preservation
- Chart generation with professional styling and 300 DPI output
- Error handling with detailed logging and recovery

### Code Quality Improvements

- All function docstrings converted to English
- Comments translated and standardized
- Variable names and constants clarified
- Professional console output without decorative elements
- Improved code organization and modularity

### Performance Optimizations

- Memory-efficient data structures
- Optimized statistical calculations
- Enhanced parallel processing efficiency
- Reduced memory footprint for large datasets

## [1.5.0] - 2025-11-10

### Added

- Advanced timeout system for all algorithms
- Timeout statistics tracking (success/failure/timeout categorization)
- Enhanced CSV exports with timeout data
- Timeout rate analysis in visualization

### Fixed

- Critical bug in GA tuple unpacking during parallel execution
- Missing timeout statistics in results dictionaries
- Inconsistent timeout handling across algorithms

### Enhanced

- Statistical analysis with separate timeout categorization
- Performance measurement using perf_counter() for nanosecond precision
- Results structure with comprehensive timeout tracking

## [1.4.0] - 2025-11-09

### Added

- Concurrent tuning mode for parallel fitness function optimization
- Advanced statistical functions with detailed metrics
- Enhanced data export with logical cost analysis
- Improved visualization with professional charts

### Enhanced

- Parallel processing architecture using ProcessPoolExecutor
- Statistical analysis with confidence intervals
- Memory management for large-scale experiments
- Error handling and logging capabilities

## [1.3.0] - 2025-11-08

### Added

- Six genetic algorithm fitness functions (F1-F6)
- Automated parameter tuning system
- Comprehensive CSV data export
- Basic visualization framework

### Enhanced

- Algorithm implementations with timeout support
- Statistical analysis capabilities
- Parallel execution framework
- Results aggregation and analysis

## [1.2.0] - 2025-11-07

### Added

- Simulated Annealing algorithm implementation
- Genetic Algorithm basic implementation
- Parameter configuration system
- Multiple independent run support

### Enhanced

- Backtracking algorithm with iterative implementation
- Performance measurement and timing
- Basic statistical analysis
- File output organization

## [1.1.0] - 2025-11-06

### Added

- Iterative Backtracking implementation
- Basic N-Queens conflict detection
- Solution validation system
- Performance timing measurements

### Enhanced

- Code structure and organization
- Error handling
- Basic logging capabilities

## [1.0.0] - 2025-11-05

### Added

- Initial project structure
- Basic N-Queens problem definition
- Core mathematical functions
- Foundation for algorithm implementations
- MIT License
- Initial README and documentation

### Core Features

- N-Queens problem representation using array format
- Efficient conflict counting using Counter data structures
- Mathematical foundation for solution validation
- Extensible architecture for multiple algorithms

---

## Version History Summary

- **2.0.0**: Complete professional English internationalization with advanced analytics
- **1.5.0**: Advanced timeout system and critical bug fixes
- **1.4.0**: Concurrent processing and enhanced statistics
- **1.3.0**: Complete fitness function suite and automated tuning
- **1.2.0**: Full algorithm implementation with SA and GA
- **1.1.0**: Backtracking implementation and performance measurement
- **1.0.0**: Initial release with core mathematical foundations

## Development Notes

### Code Quality Standards

- All code and documentation maintained in English
- Professional scientific presentation standards
- Comprehensive testing and validation
- Performance optimization and profiling
- Memory management and resource efficiency

### Future Development

- Additional metaheuristic algorithms
- Advanced visualization techniques
- Machine learning integration
- Distributed computing support
- Real-time performance monitoring

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and standards.
