# Development Guide

## Architecture Overview

SOMA is organized into several core modules:

1. **Orchestrator**: Manages the overall workflow and interactions between different components.
2. **Experience Store**: Stores and retrieves experiences for learning and decision-making.
3. **Belief Store**: Manages beliefs and their confidence levels.
4. **LLM Client**: Interfaces with large language models for various tasks.
5. **Router**: Routes tasks to the appropriate modules based on confidence levels.
6. **Curiosity Engine**: Drives the exploration of new and novel problems.
7. **Hypothesis Generator**: Generates hypotheses based on existing knowledge and data.
8. **Experiment Runner**: Executes experiments to validate hypotheses.
9. **PR Monitor**: Monitors pull requests and integrates with version control systems.
10. **Goal Store**: Manages and tracks goals and objectives.
11. **Repo Tracker**: Tracks repositories and scores new issues based on SOMA's capabilities.

## Bootstrap Process

1. **Initialization**: SOMA initializes by loading configuration settings and setting up necessary data stores.
2. **Experience and Belief Loading**: Experiences and beliefs are loaded from persistent storage.
3. **LLM Setup**: Connections to large language models are established.
4. **Routing Setup**: The router is configured to direct tasks to the appropriate modules.

## Dream Cycle 8 Steps

1. **Input Task**: Receive a task or query from the user.
2. **Confidence Assessment**: Evaluate the task based on existing beliefs and experiences.
3. **Model Selection**: Choose the appropriate model (e.g., quick model, tier 1, tier 2, tier 3) based on the confidence assessment.
4. **Execution**: Execute the task using the selected model.
5. **Verification**: Verify the results of the execution.
6. **Feedback Loop**: Update beliefs and experiences based on the verification results.
7. **Curiosity Driven Exploration**: Identify new areas for exploration and learning based on the feedback loop.
8. **Repetition**: Repeat the cycle for new tasks or until the goal is achieved.

## Contributing

Contributions are welcome! Please ensure your code adheres to the [Code of Conduct](CODE_OF_CONDUCT.md) and follows the development guidelines outlined in this document.
