import sys
sys.path.append('/data/SWATGenXApp/codes/')

from AI_agent.interactive_agent import interactive_agent

message = "explain Ingham county"
response = interactive_agent(message)
print(response)