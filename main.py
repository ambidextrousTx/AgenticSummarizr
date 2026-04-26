from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# 1. Connect to your local brain
# Replace "llama3" with whichever model you pulled (e.g., "mistral", "phi3")
local_llm = Ollama(model="llama3")

# 2. Define the Agent
# Note how we give it a "backstory" - this is the "Agentic" part.
# researcher = Agent(
#   role='Senior Systems Architect',
#   goal='Explain the shift from Imperative to Agentic programming',
#   backstory="""You are a veteran software engineer from the 90s. 
#   You explain things using analogies related to manual gearboxes and 
#   automated transmissions.""",
#   llm=local_llm,
#   allow_delegation=False,
#   verbose=True
# )
# 
# # 3. Define the Task
# test_task = Task(
#   description="Write a 2-paragraph comparison between Imperative and Agentic coding.",
#   expected_output="A concise comparison formatted as a 90s README.txt file.",
#   agent=researcher
# )
# 
# # 4. Assemble the Crew
# # Even with one agent, we wrap it in a Crew to handle the execution flow.
# crew = Crew(
#   agents=[researcher],
#   tasks=[test_task],
#   process=Process.sequential
# )
# 
# # 5. Kickoff!
# print("### Starting the Local Agent ###")
# result = crew.kickoff()
# 
# print("\n\n########################")
# print("## AGENT OUTPUT BELOW ##")
# print("########################\n")
# print(result)

# --- AGENTS ---

scout = Agent(
  role='Lead Research Scout',
  goal='Extract the 3 most impactful trends in AI from the provided context',
  backstory='You are an expert at filtering signal from noise. You find the "meat" of any topic.',
  llm=local_llm,
  verbose=True
)

writer = Agent(
  role='Technical Documentation Specialist',
  goal='Write a brief executive summary based on the scout\'s findings',
  backstory='You specialize in making complex topics digestible for busy executives.',
  llm=local_llm,
  verbose=True
)

# --- TASKS ---

task_research = Task(
  description="Analyze the current state of local LLMs in 2024.",
  expected_output="A bulleted list of 3 key technical breakthroughs.",
  agent=scout
)

task_write = Task(
  description="Using the scout's list, write a 1-paragraph executive summary.",
  expected_output="A polished, professional paragraph.",
  agent=writer
)

# --- THE CREW ---

bureau_crew = Crew(
  agents=[scout, writer],
  tasks=[task_research, task_write],
  process=Process.sequential # Task 1 -> Task 2
)

result = bureau_crew.kickoff()
print(result)
