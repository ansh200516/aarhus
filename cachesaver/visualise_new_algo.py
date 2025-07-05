import json
import matplotlib.pyplot as plt

def load_logs(run_id) -> str:
    with open(f"logs/new_algo_logs.log", 'r') as f:
        logs = f.read()
        logs = logs.split('#################################################################')
        num_logs = len(logs) - 1
        logs = logs[run_id].strip()
    return logs, num_logs


def get_widths_for_all_idxs(logs: list[str]):
    widths = {}

    for log in logs:
        log = json.loads(log)
        
        key = list(log.keys())[0]

        if 'step' not in key:
            continue

        idx = int(key.split('-')[1])

        if idx not in widths:
            widths[idx] = []
        
        log = (list(log.values())[0])
        widths[idx].append(len(log['input_states']))

    return widths



def plot_widths_for_all_idxs_with_colors(logs: list[str], _id):
    widths = {}
    colors = {}

    for log in logs:
        log = json.loads(log)
        
        key = list(log.keys())[0]

        if 'step' not in key:
            continue

        idx = int(key.split('-')[1])

        if idx not in widths:
            widths[idx] = []
            colors[idx] = 'red'
        
        log = (list(log.values())[0])
        widths[idx].append(len(log['input_states']))

        if len(log['solved_idxs']) > 0:
            colors[idx] = 'green'


    num_plots = len(widths)
    fig, axes = plt.subplots(num_plots, 1, figsize=(6, 80), sharex=True)

    for i, (idx, ws) in enumerate(widths.items()):
        axes[i].plot(range(1, 1+len(ws)), ws, color=colors[idx], marker='o')
        axes[i].set_ylabel(f"Puzzle {idx}")
    
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(f'tmp/stacked_plots_{_id}.png')





if __name__ == '__main__':
    index = -1
    logs, num_logs = load_logs(index)
    logs = logs.splitlines()
    plot_widths_for_all_idxs_with_colors(logs, _id=(index + num_logs)%num_logs)