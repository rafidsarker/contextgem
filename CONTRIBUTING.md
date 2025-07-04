# Contributing to ContextGem

Thank you for your interest in contributing to ContextGem! This document provides guidelines and instructions for contributing to the project.


## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the expectations for all interactions within the project.

## Contributor Agreement

Before your contribution can be accepted, you must sign our [Contributor Agreement](/.github/CONTRIBUTOR_AGREEMENT.md). This is a legal document that grants us the necessary rights to use your contribution.

To sign the agreement:
1. Read the [Contributor Agreement](/.github/CONTRIBUTOR_AGREEMENT.md) carefully
2. Create a copy of the agreement at `.github/contributors/[your-github-username].md`
3. Fill in all the requested information and include it with your first pull request


## Getting Started


### Development Environment

1. **Fork and clone the repository**:

    - First, fork the repository by clicking the "Fork" button on the GitHub project page

    - Then clone your fork to your local machine:

    ```bash
    git clone https://github.com/YOUR-GITHUB-USERNAME/contextgem.git
    cd contextgem
    ```
    
    - Add the original repository as an upstream remote:

    ```bash
    git remote add upstream https://github.com/shcherbak-ai/contextgem.git
    ```

2. **Set up the development environment**:
    ```bash
    # Install poetry if you don't have it
    pip install poetry

    # Install dependencies and development extras
    poetry install --with dev

    # Activate the virtual environment
    poetry shell
    ```

3. **Install pre-commit hooks**:
    ```bash
    # Install pre-commit hooks
    pre-commit install

    # Install commit-msg hooks (for commitizen)
    pre-commit install --hook-type commit-msg
    ```


### Making Changes

1. **Create a new branch**:

    For example:
    ```bash
    git checkout -b feature/your-feature-name
    ```

    When creating a branch, use one of the following prefixes that matches your change type:

    - `bugfix/` - For bug fixes (e.g., `bugfix/fix-login-timeout`)
    - `feature/` - For new features (e.g., `feature/add-dark-mode`)
    - `breaking/` - For breaking changes (e.g., `breaking/redesign-api-v2`)
    - `docs/` - For documentation updates (e.g., `docs/update-installation-guide`)
    - `perf/` - For performance improvements (e.g., `perf/optimize-database-queries`)
    - `refactor/` - For code cleanup or refactoring (e.g., `refactor/simplify-error-handling`)

    General guidelines:
    - Use hyphens (-) between words, not underscores or spaces
    - Be specific but concise about what the branch does
    - Include issue numbers when applicable (e.g., `bugfix/issue-42-user-login`)
    - Keep branch names lowercase

2. **Make your changes** following our code style guidelines.

    We use several tools to maintain code quality:

    - **Black**: For code formatting
    - **isort**: For import sorting
    - **Bandit**: For Python security vulnerability scanning
    - **Pre-commit hooks**: To automatically check and format code before commits

    The pre-commit hooks will automatically check and format your code when you commit. There are two scenarios to be aware of:

    **If the hooks modify any files during commit** (Black/isort formatting):
    1. Review the changes made by the formatters
    2. Add the modified files to the staging area
    3. Commit again

    **If security issues are detected** (Bandit):
    1. Review the security findings in the terminal output
    2. Fix the identified security issues in your code
    3. Add the fixed files to the staging area
    4. Commit again

3. **Run tests** to ensure your changes do not break existing functionality:
   ```bash
   pytest
   ```

   Please note that we use [pytest-recording](https://github.com/kiwicom/pytest-recording) to record and replay LLM API interactions. Your changes may require re-recording VCR cassettes for the tests. See [VCR Cassette Management](#vcr-cassette-management) section below for details.

4. **Commit your changes** using Conventional Commits format:
   
   We use [Conventional Commits](https://www.conventionalcommits.org/) format for our commit messages, which enables automatic changelog generation and semantic versioning. Instead of using regular git commit, please use commitizen:

   ```bash
   poetry run cz commit
   ```

   This will guide you through an interactive prompt to create a properly formatted commit message with:
   - Type of change (feat, fix, docs, style, refactor, etc.)
   - Optional scope (e.g., api, cli, docs)
   - Short description
   - Optional longer description and breaking change notes

   Example of resulting commit message:
   ```
   docs(readme): update installation instructions
   ```


## Pull Request Process

1. **Update your fork** with the latest changes from the main repository.

2. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a pull request** from your branch to the main repository.

4. **Fill out the pull request template** with all required information.

5. **Sign the Contributor Agreement** by including your filled-in `.github/contributors/[your-github-username].md` file (required with the first pull request).

6. **Wait for review**. Maintainers will review your PR and may request changes.

7. **Address review comments** if requested.


## Issues and Feature Requests

When submitting issues or feature requests, please use our GitHub issue templates:

1. Check if a similar issue already exists in the [Issues](https://github.com/shcherbak-ai/contextgem/issues) section.

2. If not, create a new issue using the appropriate template:
   - **Bug Report**: For reporting bugs or unexpected behavior
   - **Feature Request**: For suggesting new features or enhancements
   - **Documentation Improvement**: For suggesting improvements to our documentation

Each template will guide you through providing all the necessary information for your specific request.

By submitting issues or feature requests to this project, you acknowledge that these suggestions may be implemented by the project maintainers without attribution or compensation.


## Testing

- Write tests for new features or bug fixes
- Make sure all tests pass before submitting a PR
- Aim for good test coverage


### VCR Cassette Management

We use [pytest-recording](https://github.com/kiwicom/pytest-recording) to record and replay HTTP interactions with LLM APIs. This allows tests to run without making actual API calls after the initial recording.

#### When to Re-record Cassettes

You **must** re-record cassettes if:
- You modified any parameters in LLM API calls
- You're writing a new test that calls the LLM API
- The existing cassettes are no longer compatible with your changes
- You made changes to local LLM API interactions (requires Ollama and LM Studio installed on your system)

#### How to Re-record Cassettes

1. Delete the existing cassette files from the `tests/cassettes/` directory that your test uses
2. Set up the following environment variables in a `.env` file:
   ```
   CONTEXTGEM_OPENAI_API_KEY=your_openai_api_key
   CONTEXTGEM_AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   CONTEXTGEM_AZURE_OPENAI_API_BASE=your_azure_openai_base
   CONTEXTGEM_AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
   ```
   
3. Set the logger level in your `.env` file for detailed output:
   ```
   CONTEXTGEM_LOGGER_LEVEL=DEBUG
   ```
4. **For local LLM testing**: Install [Ollama](https://ollama.ai/) and [LM Studio](https://lmstudio.ai/) on your system if you're testing local model interactions

5. Run your tests, which will create new cassette files

**Important**: Re-recording cassettes will use your OpenAI API key and may incur charges to your account based on the number and type of API calls made during testing. Please be aware of these potential costs before re-recording.

Note that our VCR configuration is set up to automatically strip API keys and other personal data from the cassettes by default.


### Running Tests

Run all tests:
```bash
pytest
```

Run specific tests:
```bash
# Run a specific test method
pytest tests/test_all.py::TestAll::test_extract_all
```

### Debugging Tests

The log output will show detailed information about test execution.


## Documentation

- Update documentation for any changed functionality
- Document new features
- Use clear, concise language

### Building the Documentation

Navigate to the docs/ directory and run:

```bash
# Build with verbose output and ignore cache (recommended for structural changes)
sphinx-build -b html source build/html -v -E
```

The `-E` flag ensures Sphinx completely rebuilds the environment, which is especially important after structural changes like modifying toctree directives or removing files.

### Viewing the Documentation

After building, open `build/html/index.html` in your web browser to view the documentation.

### GitHub Pages Deployment

The documentation is automatically built and deployed to GitHub Pages when changes are pushed to the main branch. The deployment is handled by a GitHub Actions workflow (`.github/workflows/docs.yml`).

You can access the live documentation at: https://contextgem.dev

### Documentation Structure

- `source/` - Contains the source `.rst` files
- `source/_static/` - Static assets like images
- `source/conf.py` - Sphinx configuration
- `build/` - Generated documentation (not committed to version control)

### Updating README.md

The project's README.md is generated from a template. Do not modify README.md directly as your changes will be overwritten.

Instead:
1. Edit the template file at `dev/README.TEMPLATE.md`
2. The pre-commit hook will automatically update README.md using the template

If you need to test the README generation manually:

```bash
# Populate README.md from template
python dev/populate_project_readme.py
```


## Questions?

If you have any questions or need help, please reach out through:
- Opening an issue
- Contacting the maintainers at sergii@shcherbak.ai


Thank you for contributing to ContextGem!
