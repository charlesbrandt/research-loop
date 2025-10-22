# --- IMPORTS ---
import os
import yaml
import json
import subprocess
import logging
import re
import sys
import argparse
import tty
import termios
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, RootModel, Field

from crewai import Agent, Task, Crew, Process
from crewai.tasks import TaskOutput
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool, FileWriterTool

import litellm

# --- ENVIRONMENT & TELEMETRY ---
# litellm.set_verbose = True
# os.environ['LITELLM_LOG'] = 'DEBUG'
litellm._turn_on_debug()

# See also setting in .env file
# This will cause problems in subprocesses if enabled
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

# --- CONFIGURATION & CONSTANTS ---
PROJECT_CONFIG_FILE = "project_config.yaml"
PROMPT_TEMPLATE_FILE = "templates/prompts.yaml"
OUTPUT_DIR = "./output"
# Phase 1: Research Design
DESIGN_DIR = f"{OUTPUT_DIR}/1_design"

# Phase 2: Experiment Execution
EXECUTION_DIR = f"{OUTPUT_DIR}/2_execution"
EXECUTION_DATA_DIR = f"{EXECUTION_DIR}/data"

# Phase 3: Analysis
ANALYSIS_DIR = f"{OUTPUT_DIR}/3_analysis"
ANALYSIS_NOTEBOOKS_DIR = f"{ANALYSIS_DIR}/notebooks"

# Phase 4: Reporting and Dissemination
REPORTING_DIR = f"{OUTPUT_DIR}/4_reporting"
REPORTING_FIGURES_DIR = f"{REPORTING_DIR}/figures"

# Common directories
SRC_DIR = f"{OUTPUT_DIR}/src"
TESTS_DIR = f"{OUTPUT_DIR}/tests"
LOGS_DIR = f"{OUTPUT_DIR}/logs"
MAX_FIX_ATTEMPTS = 3


# --- STRUCTURED LOGGING SETUP ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        standard_attrs = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_record[key] = value

        if record.args:
            log_record["details"] = [str(arg) for arg in record.args]

        return json.dumps(log_record)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(JsonFormatter())
logger.addHandler(stdout_handler)


def setup_logger(log_file_name, append=False):
    """Configures the root logger to output to a specific file."""
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    file_mode = "a" if append else "w"
    file_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, log_file_name), mode=file_mode
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    logger.info(f"Logging reconfigured to '{log_file_name}' (mode: {file_mode})")


# Initial logger setup
setup_logger("run.log", append=False)


# --- LITELLM CALLBACK FOR DETAILED LOGGING ---
class LiteLLMLoggingCallback:
    def litellm_pre_call(self, kwargs):
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        logger.info("LLM Call Pre-Call", extra={"model": model, "messages": messages})

    def litellm_post_call(self, kwargs, response):
        model = kwargs.get("model", "unknown")
        if response is None:
            response_content = "None response"
        else:
            response_content = getattr(response, "content", str(response))
        logger.info(
            "LLM Call Post-Call", extra={"model": model, "response": response_content}
        )

    def litellm_failure_callback(self, kwargs, response):
        model = kwargs.get("model", "unknown")
        logger.error("LLM Call Failed", extra={"model": model, "error": str(response)})


# --- PYDANTIC MODELS ---
class Stories(RootModel[Dict[str, str]]):
    """Stories is a pydantic class for CrewAI to parse stories into from json output."""

    pass


class ScaffoldOutput(RootModel[Dict[str, str]]):
    """ScaffoldOutput is a pydantic class for CrewAI to parse scaffolding results into from json output."""

    root: Dict[str, str] = Field(
        ...,
        example={
            "src/my_project/main.py": "created",
            "tests/my_project/test_main.py": "created",
            "README.md": "created",
        },
    )


# --- LLM AND TOOL CONFIGURATION ---
try:
    llm = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE", "http://192.168.2.77:8080/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        temperature=0.1,
        max_tokens=32000,
    )
    logger.info("SUCCESS: ChatOpenAI client initialized successfully.")
    litellm.callbacks = [LiteLLMLoggingCallback()]
except Exception as e:
    logger.error(f"ERROR: Failed to initialize ChatOpenAI client: {e}", exc_info=True)
    sys.exit(1)


# --- CALLBACK & HELPER FUNCTIONS ---
def _parse_and_validate_output(
    output: TaskOutput, pydantic_model: BaseModel, callback_name: str
):
    """
    Helper to parse raw task output and validate with a Pydantic model.
    Handles CrewAI's output variations (json_dict, pydantic, or raw).
    """
    logger.info(f"DEBUG: {callback_name} called.")
    logger.debug(
        f"DEBUG: output.raw type: {type(output.raw)}, content (first 500 chars): {str(output.raw)[:500]}"
    )
    logger.debug(
        f"DEBUG: output.json_dict type: {type(output.json_dict)}, content: {output.json_dict}"
    )
    logger.debug(
        f"DEBUG: output.pydantic type: {type(output.pydantic)}, content: {output.pydantic}"
    )

    if output.pydantic and isinstance(output.pydantic, pydantic_model):
        logger.info(f"DEBUG: Using output.pydantic for {callback_name}.")
        return output.pydantic.root

    try:
        raw_json_data = json.loads(output.raw)
        validated_data = pydantic_model.model_validate(raw_json_data).root
        logger.info(
            f"DEBUG: Using Pydantic.model_validate(raw_json_data).root for {callback_name}."
        )
        return validated_data
    except json.JSONDecodeError as e:
        logger.error(
            f"JSON decoding error in {callback_name}: {e}",
            exc_info=True,
            extra={"raw_output": output.raw},
        )
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during Pydantic validation in {callback_name}: {e}",
            exc_info=True,
            extra={"raw_output": output.raw},
        )
        raise


def render_crew_definitions(
    project_config,
    methodology_content="",
    experiment_protocol_content="",
    experiment_results_content="",
):
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(PROMPT_TEMPLATE_FILE)
    context = {
        **project_config,
        "methodology_content": methodology_content,
        "experiment_protocol_content": experiment_protocol_content,
        "experiment_results_content": experiment_results_content,
    }
    rendered_yaml_str = template.render(context)
    return yaml.safe_load(rendered_yaml_str)


def create_project_directories(project_dir):
    # Phase 1: Research Design
    os.makedirs(DESIGN_DIR, exist_ok=True)

    # Phase 2: Experiment Execution
    os.makedirs(EXECUTION_DATA_DIR, exist_ok=True)

    # Phase 3: Analysis
    os.makedirs(ANALYSIS_NOTEBOOKS_DIR, exist_ok=True)

    # Phase 4: Reporting and Dissemination
    os.makedirs(REPORTING_FIGURES_DIR, exist_ok=True)

    # Common directories
    os.makedirs(SRC_DIR, exist_ok=True)
    os.makedirs(TESTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


# --- PHASE-SPECIFIC FUNCTIONS ---
def run_research_design_crew_tasks(project_config):
    """Encapsulates the tasks for the research design crew."""
    crew_defs = render_crew_definitions(project_config)
    agents = {
        name: Agent(llm=llm, **props) for name, props in crew_defs["agents"].items()
    }

    literature_review_task = Task(
        agent=agents["LiteratureReviewer"],
        name="Conduct Literature Review",
        **crew_defs["tasks"]["conduct_literature_review"],
    )

    hypothesis_file_path = os.path.join(DESIGN_DIR, "HYPOTHESIS.md")
    hypothesis_task = Task(
        agent=agents["HypothesisGenerator"],
        name="Generate Research Hypothesis",
        context=[literature_review_task],
        output_file=hypothesis_file_path,
        **crew_defs["tasks"]["generate_hypothesis"],
    )

    methodology_file_path = os.path.join(DESIGN_DIR, "METHODOLOGY.md")
    methodology_task = Task(
        agent=agents["MethodologyDesigner"],
        name="Design Research Methodology",
        context=[hypothesis_task],
        output_file=methodology_file_path,
        **crew_defs["tasks"]["design_methodology"],
    )

    research_design_crew = Crew(
        agents=[
            agents["LiteratureReviewer"],
            agents["HypothesisGenerator"],
            agents["MethodologyDesigner"],
        ],
        tasks=[literature_review_task, hypothesis_task, methodology_task],
        process=Process.sequential,
        verbose=True,
        share_crew=False,
        human_input=False,
        telemetry=False,
        tracing=False,
        output_log_file=os.path.join(LOGS_DIR, "research_design_crew.log"),
    )
    research_design_crew.kickoff()
    logger.info(f"Research design complete. Documents saved in '{DESIGN_DIR}'.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the AI developer workflow.")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically approve the planning phase",
    )
    parser.add_argument(
        "--skip-planning",
        action="store_true",
        help="Skip the planning phase and proceed directly to development",
    )
    return parser.parse_args()


def load_project_config():
    with open(PROJECT_CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def setup_project_environment(project_config):
    project_dir_name = project_config["project_name"].lower().replace(" ", "_")
    create_project_directories(project_dir_name)


# --- PHASE-SPECIFIC FUNCTIONS ---
def handle_research_design_phase(project_config, args):
    logger.info("--- PHASE 1: RESEARCH DESIGN ---")
    hypothesis_content = ""
    methodology_content = ""

    if not args.skip_planning:
        logger.info("Starting research design phase...")
        run_research_design_crew_tasks(project_config)

        hypothesis_path = os.path.join(DESIGN_DIR, "HYPOTHESIS.md")
        methodology_path = os.path.join(DESIGN_DIR, "METHODOLOGY.md")

        if not os.path.exists(hypothesis_path) or not os.path.exists(methodology_path):
            logger.error(
                "Research design documents were not created properly. Exiting."
            )
            sys.exit(1)

        logger.info(f"Research design complete. Documents saved in '{DESIGN_DIR}'.")

        if args.yes:
            logger.info("Auto-approving the research design phase.")
        else:
            print(
                "Please review research design documents. Press ENTER to continue or Ctrl+C to exit."
            )
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch != "\r":  # Check for Enter key
                    logger.info("Research design not approved. Exiting.")
                    sys.exit(0)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
        logger.info("Skipping research design phase as requested.")

    hypothesis_path = os.path.join(DESIGN_DIR, "HYPOTHESIS.md")
    methodology_path = os.path.join(DESIGN_DIR, "METHODOLOGY.md")

    if os.path.exists(hypothesis_path):
        with open(hypothesis_path, "r") as f:
            hypothesis_content = f.read()
    if os.path.exists(methodology_path):
        with open(methodology_path, "r") as f:
            methodology_content = f.read()

    return hypothesis_content, methodology_content


def handle_experimentation_phase(project_config, methodology_content):
    logger.info("--- PHASE 2: EXPERIMENTATION ---")
    setup_logger("experimentation.log")
    logger.info("Starting Experimentation phase...")

    crew_defs = render_crew_definitions(
        project_config, methodology_content=methodology_content
    )
    agents = {
        name: Agent(llm=llm, **props) for name, props in crew_defs["agents"].items()
    }

    # Step 1: Experiment Designer creates the experiment protocol
    experiment_protocol_file_path = os.path.join(DESIGN_DIR, "EXPERIMENT_PROTOCOL.md")
    experiment_protocol_task = Task(
        agent=agents["ExperimentDesigner"],
        name="Design Experiment Protocol",
        description=crew_defs["tasks"]["design_experiment_protocol"][
            "description"
        ].format(methodology_content=methodology_content),
        expected_output=crew_defs["tasks"]["design_experiment_protocol"][
            "expected_output"
        ],
        output_file=experiment_protocol_file_path,
    )

    experiment_design_crew = Crew(
        agents=[agents["ExperimentDesigner"]],
        tasks=[experiment_protocol_task],
        share_crew=False,
        human_input=False,
        verbose=False,
        telemetry=False,
        tracing=False,
    )
    experiment_design_crew.kickoff()

    experiment_protocol_content = ""
    if os.path.exists(experiment_protocol_file_path):
        with open(experiment_protocol_file_path, "r") as f:
            experiment_protocol_content = f.read()
    else:
        logger.error("Experiment protocol was not created. Exiting.")
        sys.exit(1)

    logger.info("Experimentation phase completed successfully.")
    setup_logger("run.log", append=True)
    return experiment_protocol_content


def handle_experiment_execution_and_analysis_phase(
    project_config, experiment_protocol_content
):
    logger.info("--- PHASE 3: EXPERIMENT EXECUTION AND ANALYSIS ---")
    setup_logger("execution_analysis.log")
    logger.info("Starting experiment execution and analysis phase...")

    crew_defs = render_crew_definitions(
        project_config, experiment_protocol_content=experiment_protocol_content
    )
    agents = {
        name: Agent(llm=llm, **props) for name, props in crew_defs["agents"].items()
    }

    # Step 1: Experiment Conductor executes the protocol
    experiment_results_file_path = os.path.join(EXECUTION_DIR, "EXPERIMENT_RESULTS.md")
    conduct_experiment_task = Task(
        agent=agents["ExperimentConductor"],
        name="Conduct Experiment",
        description=crew_defs["tasks"]["conduct_experiment"]["description"].format(
            experiment_protocol_content=experiment_protocol_content
        ),
        expected_output=crew_defs["tasks"]["conduct_experiment"]["expected_output"],
        output_file=experiment_results_file_path,
    )

    experiment_execution_crew = Crew(
        agents=[agents["ExperimentConductor"]],
        tasks=[conduct_experiment_task],
        share_crew=False,
        human_input=False,
        verbose=False,
        telemetry=False,
        tracing=False,
    )
    experiment_execution_crew.kickoff()

    experiment_results_content = ""
    if os.path.exists(experiment_results_file_path):
        with open(experiment_results_file_path, "r") as f:
            experiment_results_content = f.read()
    else:
        logger.error("Experiment results were not created. Exiting.")
        sys.exit(1)

    # Step 2: Data Analyzer analyzes the results
    analysis_report_file_path = os.path.join(ANALYSIS_DIR, "ANALYSIS_REPORT.md")
    analyze_data_task = Task(
        agent=agents["DataAnalyzer"],
        name="Analyze Experiment Data",
        description=crew_defs["tasks"]["analyze_data"]["description"].format(
            experiment_results_content=experiment_results_content
        ),
        expected_output=crew_defs["tasks"]["analyze_data"]["expected_output"],
        output_file=analysis_report_file_path,
    )

    data_analysis_crew = Crew(
        agents=[agents["DataAnalyzer"]],
        tasks=[analyze_data_task],
        share_crew=False,
        human_input=False,
        verbose=False,
        telemetry=False,
        tracing=False,
    )
    data_analysis_crew.kickoff()

    logger.info("Experiment execution and analysis phase completed successfully.")
    setup_logger("run.log", append=True)
    return experiment_results_content


def handle_reporting_and_dissemination_phase(
    project_config, experiment_results_content
):
    logger.info("--- PHASE 4: REPORTING AND DISSEMINATION ---")
    setup_logger("reporting_dissemination.log")

    crew_defs = render_crew_definitions(
        project_config, experiment_results_content=experiment_results_content
    )
    agents = {
        name: Agent(llm=llm, **props) for name, props in crew_defs["agents"].items()
    }

    # Step 1: Reporter writes the research report
    research_report_file_path = os.path.join(REPORTING_DIR, "RESEARCH_REPORT.md")
    write_research_report_task = Task(
        agent=agents["Reporter"],
        name="Write Research Report",
        description=crew_defs["tasks"]["write_research_report"]["description"],
        expected_output=crew_defs["tasks"]["write_research_report"]["expected_output"],
        output_file=research_report_file_path,
    )

    reporting_crew = Crew(
        agents=[agents["Reporter"]],
        tasks=[write_research_report_task],
        share_crew=False,
        human_input=False,
        verbose=False,
        telemetry=False,
        tracing=False,
    )
    reporting_crew.kickoff()

    # Step 2: Knowledge Disseminator creates the dissemination plan
    dissemination_plan_file_path = os.path.join(REPORTING_DIR, "DISSEMINATION_PLAN.md")
    create_dissemination_plan_task = Task(
        agent=agents["KnowledgeDisseminator"],
        name="Create Dissemination Plan",
        description=crew_defs["tasks"]["create_dissemination_plan"]["description"],
        expected_output=crew_defs["tasks"]["create_dissemination_plan"][
            "expected_output"
        ],
        output_file=dissemination_plan_file_path,
    )

    dissemination_crew = Crew(
        agents=[agents["KnowledgeDisseminator"]],
        tasks=[create_dissemination_plan_task],
        share_crew=False,
        human_input=False,
        verbose=False,
        telemetry=False,
        tracing=False,
    )
    dissemination_crew.kickoff()

    logger.info("Reporting and dissemination phase completed successfully.")
    setup_logger("run.log", append=True)
    return research_report_file_path, dissemination_plan_file_path


# --- MAIN ORCHESTRATION ---
def main():
    args = parse_arguments()
    logger.info("--- Starting Research Loop ---")
    project_config = load_project_config()
    setup_project_environment(project_config)

    _, methodology_content = handle_research_design_phase(project_config, args)
    experiment_protocol_content = handle_experimentation_phase(
        project_config, methodology_content
    )

    experiment_results_content = handle_experiment_execution_and_analysis_phase(
        project_config, experiment_protocol_content
    )

    if not experiment_results_content:
        logger.error("Experiment results content is empty. Exiting.")
        sys.exit(1)

    handle_reporting_and_dissemination_phase(project_config, experiment_results_content)

    logger.info("--- Workflow complete. ---")


if __name__ == "__main__":
    main()
