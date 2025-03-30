from clearml import Task


# task = Task.init(
#     project_name='example',    # project name of at least 3 characters
#     task_name='task template', # task name of at least 3 characters
#     task_type=None,
#     tags=None,
#     reuse_last_task_id=True,
#     continue_last_task=False,
#     output_uri=None,
#     auto_connect_arg_parser=True,
#     auto_connect_frameworks=True,
#     auto_resource_monitoring=True,
#     auto_connect_streams=True,
# )

Task.init(project_name='clearML-demo-project', task_name='clearML-demo-task')

