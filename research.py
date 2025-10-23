from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
#from crewai_tools import EXASearchTool, FirecrawlScrapeWebsiteTool
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv
import os
import sys
import logging
from datetime import datetime

load_dotenv()

# --- LLM AND TOOL CONFIGURATION ---
try:
    llm = LLM(
        base_url="http://192.168.2.77:8080/v1",
        model="gpt-4", # Placeholder, as it requires a model name
        api_key="not-needed",
        temperature=0.1,
        max_tokens=32000,
    )
    # Logging is not fully set up in research.py, so a simple print for now.
    print("SUCCESS: CrewAI LLM client initialized successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize CrewAI LLM client: {e}")
    sys.exit(1)

# Initialize tools
# For search, I will use Tavily Search API. I'll use `SerperDevTool` as a placeholder
# and assume it can be configured to use Tavily or a similar service later.
# The user will need to provide their API key.
# For reading websites, I will use `WebsiteReadTool`.

# Note: In a real-world scenario, you would configure these tools more specifically.
# For example, SerperDevTool would require SERPER_API_KEY.
# For this demonstration, we'll assume the necessary API keys are in the .env file.
# The user will be responsible for setting up the environment variables.

search_tool = SerperDevTool()  # Placeholder for a robust search API tool
web_read_tool = ScrapeWebsiteTool()
# TODO: explore:
# search_tool = EXASearchTool()
# web_read_tool = FirecrawlScrapeWebsiteTool()


# Define the Agents
research_planner_agent = Agent(
    llm=llm,
    role="Research Planner",
    goal="To take the user's high-level research topic and break it down into a structured, actionable research plan.",
    backstory="An experienced principal investigator who excels at defining research questions, identifying keywords, and structuring a study.",
    verbose=True,
    allow_delegation=False,
)

literature_searcher_agent = Agent(
    llm=llm,
    role="Literature Searcher",
    goal="To find relevant scientific papers, articles, and pre-prints from online sources.",
    backstory="A specialist in information retrieval who knows exactly how to query academic search engines to find the most impactful literature.",
    tools=[search_tool],
    verbose=True,
    allow_delegation=True,
)

data_synthesizer_agent = Agent(
    llm=llm,
    role="Data Synthesizer",
    goal="To read the collected papers, extract key findings, methodologies, and data, and structure this information for review.",
    backstory="A meticulous post-doc researcher who can quickly process dense information and synthesize it into a structured format.",
    tools=[
        web_read_tool
    ],  # Assuming web_read_tool can read the content of found papers
    verbose=True,
    allow_delegation=True,
)

critical_analyst_agent = Agent(
    llm=llm,
    role="Critical Analyst",
    goal="To analyze the synthesized data, identify contradictions between sources, point out research gaps, and question assumptions.",
    backstory="A seasoned professor known for their rigorous, critical eye and ability to find weaknesses in any argument.",
    verbose=True,
    allow_delegation=True,
)

report_writer_agent = Agent(
    llm=llm,
    role="Report Writer",
    goal="To compile all the findings, analyses, and critiques into a final, polished, and human-readable report.",
    backstory="A scientific journalist who excels at turning complex technical information into a clear and compelling narrative.",
    verbose=True,
    allow_delegation=False,
)


# Define the Tasks
plan_research_task = Task(
    description=(
        "Break down the research topic '{research_topic}' into key questions, "
        "search terms, and a high-level plan for literature review. "
        "The output should be a structured research plan."
    ),
    expected_output="A structured research plan including key questions, search terms, and a plan for literature review.",
    agent=research_planner_agent,
)

find_literature_task = Task(
    description=(
        "Using the provided research plan, conduct a thorough literature search for scientific papers, articles, "
        "and pre-prints. Focus on recent and highly cited works relevant to the topic. "
        "Compile a list of URLs/DOIs of the most relevant papers (up to 5-7 papers)."
    ),
    expected_output="A list of URLs/DOIs of 5-7 relevant scientific papers related to the research plan.",
    agent=literature_searcher_agent,
)

synthesize_data_task = Task(
    description=(
        "For each paper found, read its content and extract the following: "
        "1. Main objectives/hypotheses. "
        "2. Key methodologies used. "
        "3. Principal findings/results. "
        "4. Conclusions and implications. "
        "5. Any limitations mentioned. "
        "Consolidate this information into a structured summary for each paper."
    ),
    expected_output="A structured summary for each of the provided scientific papers, detailing objectives, methodologies, findings, conclusions, and limitations.",
    agent=data_synthesizer_agent,
    context=[
        find_literature_task
    ],  # This task depends on the output of find_literature_task
)

analyze_critique_task = Task(
    description=(
        "Review the structured summaries of the scientific papers. "
        "Identify common themes, conflicting findings, research gaps, "
        "and any potential biases or weaknesses in the methodologies. "
        "Provide a critical analysis of the current state of research."
    ),
    expected_output="A critical analysis highlighting common themes, contradictions, research gaps, and methodological weaknesses across the reviewed papers.",
    agent=critical_analyst_agent,
    context=[
        synthesize_data_task
    ],  # This task depends on the output of synthesize_data_task
)

write_report_task = Task(
    description=(
        "Based on the critical analysis and synthesized data, "
        "write a comprehensive and well-structured research report. "
        "The report should include an introduction, summary of findings, "
        "critical analysis, identified gaps, and a conclusion. "
        "The report should be suitable for a scientific audience."
    ),
    expected_output="A comprehensive scientific research report suitable for a scientific audience, incorporating all findings and critical analysis.",
    agent=report_writer_agent,
    context=[
        analyze_critique_task
    ],  # This task depends on the output of analyze_critique_task
    output_file="report.md", # This will be overwritten dynamically
)

# Build the Crew.
crew = Crew(
    agents=[
        research_planner_agent,
        literature_searcher_agent,
        data_synthesizer_agent,
        critical_analyst_agent,
        report_writer_agent,
    ],
    tasks=[
        plan_research_task,
        find_literature_task,
        synthesize_data_task,
        analyze_critique_task,
        write_report_task,
    ],
    verbose=True,
    process=Process.sequential,
)

if __name__ == "__main__":
    research_topic = input("Enter your research topic: ")
    print("########################")
    print(f"Starting CrewAI research for: {research_topic}")
    print("########################")

    # Generate a timestamped filename for the output report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output/{research_topic.replace(' ', '_')}_{timestamp}_report.md"

    # Pass the dynamic output filename to the write_report_task
    crew.tasks[-1].output_file = output_filename  # Assuming write_report_task is the last task

    result = crew.kickoff(inputs={"research_topic": research_topic})

    print("\n\n########################")
    print("## Result Details ##")
    print("########################\n")

    # Access and print individual task outputs
    for i, task in enumerate(crew.tasks):
        print(f"--- Task {i+1}: {task.description} ---")
        # Ensure task.output is not None and handle non-string outputs explicitly
        if task.output:
            print(task.output.raw_output if hasattr(task.output, 'raw_output') else str(task.output))
        else:
            print("No output available for this task.")
        print("\n")

    print(f"\n\n######### Final Report #########\n")
    print(result)
    print(f"\n\nFinal report saved to: {output_filename}")
