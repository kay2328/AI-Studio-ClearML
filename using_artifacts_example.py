# Using artifacts example
"""
Upload artifacts from a Task, and then a different Task can access and utilize the data from that artifact.
"""
from clearml import Task
from time import sleep
import pandas as pd

"Loading Models from a Task"
prev_task = Task.get_task(task_id='8147177d353a4d23bf60c33c34663a36')
last_snapshot = prev_task.models['output'][-1]
local_weights_path = last_snapshot.get_local_copy()


"use some other task's artifacts"
task2 = Task.init(project_name='demoClearML', task_name='Use artifact from other task')

# get instance of task that created artifact, using task ID
preprocess_task = Task.get_task(task_id='8147177d353a4d23bf60c33c34663a36')
# access artifact
local_csv = preprocess_task.artifacts['data'].get_local_copy()
# Read the CSV file into a DataFrame
df = pd.read_csv(local_csv)

# Print the DataFrame
print(df)

# Simulate the work of a Task
sleep(1.0)
print('Finished doing stuff with some data :)')
