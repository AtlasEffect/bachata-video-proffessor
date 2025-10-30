# Contributing to Bachata Video Professor

Thank you for your interest in contributing to this project! This document provides guidelines for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Development Setup

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/bachata-video-professor.git
   cd bachata-video-professor
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .[dev]
   ```

5. Run the tests to ensure everything is working:
   ```bash
   pytest
   ```

## Code Style

This project uses the following tools to maintain code quality:

- **Black**: For code formatting
- **Flake8**: For linting
- **MyPy**: For type checking

Please ensure your code passes all these checks before submitting a pull request:

```bash
black src tests
flake8 src tests
mypy src
pytest
```

## Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src/bachata_analyzer
```

Run specific test file:
```bash
pytest tests/test_segmentation.py
```

## Project Structure

```
bachata-video-professor/
├── src/bachata_analyzer/          # Main package
│   ├── __init__.py
│   ├── analyzer.py                # Main analyzer class
│   ├── cli.py                     # Command-line interface
│   ├── config.py                  # Configuration management
│   ├── models.py                  # Data models
│   ├── pose_detector.py           # Pose detection and tracking
│   ├── segmentation.py            # Dance segmentation
│   ├── video_processor.py         # Video handling
│   └── output_generator.py        # Output generation
├── tests/                         # Test suite
├── examples/                      # Example videos and configs
├── notebooks/                     # Jupyter notebooks
├── .github/workflows/             # GitHub Actions
├── requirements.txt               # Dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # Project documentation
```

## Contributing Guidelines

### Bug Reports

When filing a bug report, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment information (OS, Python version, etc.)
- Any relevant logs or error messages

### Feature Requests

For feature requests, please provide:

- Clear description of the proposed feature
- Use case and motivation
- Any implementation ideas (optional)

### Pull Requests

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines

3. Add tests for new functionality

4. Update documentation if needed

5. Commit your changes with clear, descriptive messages

6. Push to your fork and create a pull request

### Pull Request Guidelines

- Keep pull requests focused on a single feature or bugfix
- Ensure all tests pass
- Update documentation for any API changes
- Follow the existing code style
- Add tests for new functionality

## Development Tips

### Testing with Different Videos

You can test the pipeline with different YouTube videos:

```bash
bachata-analyze "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --out test_output/
```

### Performance Optimization

When working on performance improvements:

1. Use the example videos for benchmarking
2. Profile CPU usage with different configurations
3. Test on different hardware configurations
4. Consider memory usage for long videos

### Adding New Features

When adding new features:

1. Update the configuration model if needed
2. Add appropriate tests
3. Update the CLI if it adds new options
4. Document the feature in README
5. Consider adding examples to the notebook

## Questions?

Feel free to open an issue for questions or discussion about potential contributions.

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.