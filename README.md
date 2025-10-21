# Research Loop

This repository contains an autonomous AI research system built using CrewAI. It can take a high-level research objective and execute a comprehensive research workflow, from literature review and hypothesis generation to experiment design, execution, data analysis, and report generation.

## Architecture

This system implements an agentic workflow structured into four main phases:

1.  **Configuration (`project_config.yaml`):** You define the research specifics (project name, description, technical stack, research objectives) here. The agent prompts are generic templates, designed to adapt to the provided configuration.
2.  **Orchestration (`run.py`):** The main script that manages the entire research process. It renders prompts and orchestrates the four distinct phases.

## Research Workflow Phases

### Phase 1: Research Design

*   **Literature Reviewer Agent:** Conducts a comprehensive literature review based on the research topic and objectives, identifying knowledge gaps.
*   **Hypothesis Generator Agent:** Formulates clear, testable research hypotheses based on the literature review.
*   **Methodology Designer Agent:** Designs a detailed research methodology to test the generated hypotheses, specifying procedures, experimental setup, data collection, and analysis plans.
*   **Human-in-the-Loop:** The script pauses for your approval of the research design documents (Literature Review, Hypothesis, Methodology) before proceeding.

### Phase 2: Experimentation

*   **Experiment Designer Agent:** Creates detailed, robust experimental protocols based on the approved methodology, defining precise steps, materials, controls, and success/failure metrics.

### Phase 3: Experiment Execution and Analysis

*   **Experiment Conductor Agent:** Executes the experimental protocol, collects data, and provides raw observations.
*   **Data Analyzer Agent:** Analyzes the collected data, identifies patterns, draws conclusions, and proposes further research.

### Phase 4: Reporting and Dissemination

*   **Reporter Agent:** Synthesizes research findings, methodology, and experimental results into a comprehensive, publication-ready research report.
*   **Knowledge Disseminator Agent:** Develops a plan for effectively communicating the research findings to relevant scientific communities and stakeholders.

## Setup Instructions

### 0. Copy / Clone this project

Each new research project can start as a copy. Then edit the `project_config.yaml` as needed.

### 1. Prerequisites
- Docker installed and running.

### How to Run 

Configure Your Project: 

  - Open and edit `project_config.yaml`.
  - Define your `project_name`, `one_liner_description`, `technical_stack`, `research_objectives`, etc. This is the primary file you need to adapt for a new research project.

To run the full research workflow:

```bash
docker-compose build main && docker-compose run main python run.py
```

To bring up the Docker container and run interactively:

```bash
docker-compose up --build
docker-compose exec main bash
python run.py
```

If you need to skip the research design phase (e.g., if you're restarting after a manual approval or have pre-defined design documents):

```bash
python run.py --skip-planning
```

The agent will now begin the research process. You will be prompted in your terminal to approve steps along the way.

