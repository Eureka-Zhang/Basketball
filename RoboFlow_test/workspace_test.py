from roboflow import Roboflow
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# Correctly access and iterate over workspaces
workspace = rf.workspace()
print("Available workspaces:")
print(workspace.name)

projects = workspace.projects()
print("Available projects in workspace:")
for project in projects:
    print(project.name)