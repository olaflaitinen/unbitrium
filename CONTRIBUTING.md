# Contributing to Unbitrium

Thank you for your interest in contributing to Unbitrium! We welcome contributions from the community to help make this the standard simulator for Federated Learning research.

## Project Lead
Unbitrium is developed and maintained by **Olaf Yunus Laitinen Imanov** (<oyli@dtu.dk>).

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/unbitrium.git
    cd unbitrium
    ```
3.  **Install dependencies** (requires Python 3.14+):
    ```bash
    pip install -e ".[dev]"
    ```
4.  **Create a branch** for your feature or fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```

## Code Standards

*   **Style**: We follow strict Google python style guidelines.
*   **Documentation**: All public API methods must have docstrings in Google format.
*   **Formatting**: Use `black` and `isort` for formatting.
*   **Type Hinting**: All code must be fully type-hinted and pass `mypy --strict`.
*   **Testing**: New features must include unit tests (pytest).

## Pull Request Process

1.  Ensure all tests pass locally: `pytest`.
2.  Update documentation if applicable.
3.  Submit a Pull Request to the `main` branch.
4.  The project lead will review your contribution.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## License

By contributing, you agree that your contributions will be licensed under the [EUPL-1.2](LICENSE) License.
