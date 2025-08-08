def create_groups(students, group_size):
    return [students[i:i + group_size] for i in range(0, len(students), group_size)]
