def search_logs(log_file, target_logs):
    found_logs = {log: [] for log in target_logs}
    with open(log_file, 'r') as file:
        for line in file:
            for target_log in target_logs:
                if target_log in line:
                    found_logs[target_log].append(line)
    return found_logs

def check_pymc_logs(log_file):
    status = '5004'
    target_logs = ['5002', '5003']
    found_logs = search_logs(log_file, target_logs)
    if found_logs['5002']:
        status =  '5002'
    elif found_logs['5003']:
        status =  '5003'
    else:
        status =  '5004'
    return status