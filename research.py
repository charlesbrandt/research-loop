from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteReadTool
from dotenv import load_dotenv
import os

load_dotenv()

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
web_read_tool = WebsiteReadTool()


# Define the Agents
research_planner_agent = Agent(
    role="Research Planner",
    goal="To take the user's high-level research topic and break it down into a structured, actionable research plan.",
    backstory="An experienced principal investigator who excels at defining research questions, identifying keywords, and structuring a study.",
    verbose=True,
    allow_delegation=False,
)

literature_searcher_agent = Agent(
    role="Literature Searcher",
    goal="To find relevant scientific papers, articles, and pre-prints from online sources.",
    backstory="A specialist in information retrieval who knows exactly how to query academic search engines to find the most impactful literature.",
    tools=[search_tool],
    verbose=True,
    allow_delegation=True,
)

data_synthesizer_agent = Agent(
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
    role="Critical Analyst",
    goal="To analyze the synthesized data, identify contradictions between sources, point out research gaps, and question assumptions.",
    backstory="A seasoned professor known for their rigorous, critical eye and ability to find weaknesses in any argument.",
    verbose=True,
    allow_delegation=True,
)

report_writer_agent = Agent(
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
    verbose=2,
    process=Process.sequential,
)

if __name__ == "__main__":
    # You can change the research topic here
    research_topic = input("Enter your research topic: ")
    print("########################")
    print(f"Starting CrewAI research for: {research_topic}")
    print("########################")
    result = crew.kickoff(inputs={"research_topic": research_topic})
    print("\n\n########################")
    print("## Research Report ##")
    print("########################\n")
    print(result)
