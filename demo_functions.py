import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from clearml import Task, Dataset

# create a task for the data processing
task = Task.init(project_name='demoClearML', task_name='dataset-create', task_type='data_processing')

# get the v1 dataset
dataset = Dataset.get(dataset_project='demoClearML', dataset_name='dataset_v1')

# get a local mutable copy of the dataset
dataset_folder = dataset.get_mutable_local_copy(
    target_folder='work_dataset',
    overwrite=True
)

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
# X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

X_train = torch.tensor(pd.read_csv('work_dataset/X_train.csv').values, dtype=torch.float32)
y_train = torch.tensor(pd.read_csv('work_dataset/y_train.csv').values.flatten(), dtype=torch.long)
X_test = torch.tensor(pd.read_csv('work_dataset/X_test.csv').values, dtype=torch.float32)
y_test = torch.tensor(pd.read_csv('work_dataset/y_test.csv').values.flatten(), dtype=torch.long)

import pdb; pdb.set_trace()

# Add a README file to the work_dataset folder
with open('work_dataset/README.md', 'w') as f:
    f.write('# Work Dataset\n\nThis folder contains the mutable local copy of the dataset used for data processing.')

# create a new version of the dataset with the pickle file
new_dataset = Dataset.create(
    dataset_project='data',
    dataset_name='dataset_v2',
    parent_datasets=[dataset],
    # this will make sure we have the creation code and the actual dataset artifacts on the same Task
    use_current_task=True,
)
new_dataset.sync_folder(local_path='work_dataset')
new_dataset.upload()
new_dataset.finalize()

# now let's remove the previous dataset tag
dataset.tags = []
new_dataset.tags = ['latest']

# # Save the preprocessed data as CSV
# pd.DataFrame(X_train.numpy()).to_csv('preprocessed_data/X_train.csv', index=False)
# pd.DataFrame(y_train.numpy()).to_csv('preprocessed_data/y_train.csv', index=False)
# pd.DataFrame(X_test.numpy()).to_csv('preprocessed_data/X_test.csv', index=False)
# pd.DataFrame(y_test.numpy()).to_csv('preprocessed_data/y_test.csv', index=False)

# task.upload_artifact(name='data', artifact_object='preprocessed_data/X_train.csv')

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the neural network, loss function, and optimizer
net = Net()

"Loading Models from a Task"
prev_task = Task.get_task(task_id='8147177d353a4d23bf60c33c34663a36')
last_snapshot = prev_task.models['output'][-1]
local_weights_path = last_snapshot.get_local_copy()
net.load_state_dict(torch.load(local_weights_path))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Train the neural network
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Make predictions and print the classification report
with torch.no_grad():
    y_pred = net(X_test).argmax(dim=1)
print(classification_report(y_test, y_pred))

# Save the trained model
torch.save(net.state_dict(), 'model.pth')
task.upload_artifact(name='model', artifact_object='model.pth')
