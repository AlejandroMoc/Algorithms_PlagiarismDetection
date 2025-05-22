class Task:
    def __init__(self, code, deadline, gain):
        self.code = code  # Task ID
        self.deadline = deadline  # Deadline for task completion
        self.gain = gain  # Gain if task is completed before or on deadline

def analyze(task):
    return task.gain

def smallest_number(num1, num2):
    return num2 if num1 > num2 else num1

def task_assignment(tasks, n):
    # Sort tasks by gain in descending order
    tasks.sort(key=analyze, reverse=True)

    # List to store the result (sequence of task IDs)
    result = [-1] * n
    # Boolean list to keep track of occupied time slots
    time_slot = [False] * n

    # Iterate through all tasks
    for i in range(n):
        # Find a free slot for this task, checking from the last possible slot
        for j in range(smallest_number(n, tasks[i].deadline) - 1, -1, -1):
            if not time_slot[j]:  # If slot is free
                result[j] = i  # Assign this task to the slot
                time_slot[j] = True  # Mark the slot as occupied
                break

    # Get the sequence of task IDs for maximum gain
    task_sequence = [tasks[result[i]].code for i in range(n) if time_slot[i]]
    return task_sequence

# Example
if __name__ == "__main__":
    n = 4
    tasks = [
        Task('x', 4, 20),
        Task('y', 1, 10),
        Task('z', 1, 40),
        Task('w', 1, 30)
    ]

    # Display the sequence of tasks that maximize gain
    print("Following is the maximum gain sequence of tasks:")
    print(" ".join(task_assignment(tasks, n)))