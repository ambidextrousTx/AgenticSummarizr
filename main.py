from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# 1. Connect to your local brain
# Replace "llama3" with whichever model you pulled (e.g., "mistral", "phi3")
local_llm = Ollama(model="llama3")

# 2. Define the Agent
# Note how we give it a "backstory" - this is the "Agentic" part.
researcher = Agent(
  role='Senior Systems Architect',
  goal='Explain the shift from Imperative to Agentic programming',
  backstory="""You are a veteran software engineer from the 90s. 
  You explain things using analogies related to manual gearboxes and 
  automated transmissions.""",
  llm=local_llm,
  allow_delegation=False,
  verbose=True
)

# 3. Define the Task
test_task = Task(
  description="Write a 2-paragraph comparison between Imperative and Agentic coding.",
  expected_output="A concise comparison formatted as a 90s README.txt file.",
  agent=researcher
)

# 4. Assemble the Crew
# Even with one agent, we wrap it in a Crew to handle the execution flow.
crew = Crew(
  agents=[researcher],
  tasks=[test_task],
  process=Process.sequential
)

# 5. Kickoff!
print("### Starting the Local Agent ###")
result = crew.kickoff()

print("\n\n########################")
print("## AGENT OUTPUT BELOW ##")
print("########################\n")
print(result)
