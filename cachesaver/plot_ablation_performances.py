import matplotlib.pyplot as plt
import numpy as np

def plot_algo_bars(algorithms, env_data_dict, filename='tmp/algorithm_comparison.png'):
    env_names = list(env_data_dict.keys())
    num_envs = len(env_names)
    num_algos = len(algorithms)
    width = 0.35

    # Adjusted: More height per environment
    fig_height = 7 * num_envs  # Increased from 6 to 7
    fig_width = max(10, num_algos * 1.2)
    fig, axes = plt.subplots(num_envs, 1, figsize=(fig_width, fig_height), sharex=True)

    if num_envs == 1:
        axes = [axes]

    for idx, env in enumerate(env_names):
        ax1 = axes[idx]
        ax2 = ax1.twinx()
        data = env_data_dict[env]

        acc_avg = [d[0] for d in data]
        acc_std = [d[1] for d in data]
        cost_avg = [d[2] for d in data]
        cost_std = [d[3] for d in data]
        ind = np.arange(num_algos)

        ax1.bar(ind - width/2, acc_avg, width, yerr=acc_std, color='blue', capsize=5, label='Accuracy')
        ax2.bar(ind + width/2, cost_avg, width, yerr=cost_std, color='red', capsize=5, label='Cost')

        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, max([avg + std for avg, std in zip(cost_avg, cost_std)]) * 1.1)

        ax1.set_ylabel('Accuracy', color='blue')
        ax2.set_ylabel('Cost', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')

        ax1.set_title(f'Environment: {env}', fontsize=14, pad=20)

        # Horizontal reference lines for first algorithm
        acc0_avg, acc0_std = acc_avg[0], acc_std[0]
        cost0_avg, cost0_std = cost_avg[0], cost_std[0]

        ax1.axhline(acc0_avg, color='blue', linestyle='-', linewidth=1.5)
        ax1.axhline(acc0_avg + acc0_std, color='blue', linestyle='dotted', linewidth=1)
        ax1.axhline(acc0_avg - acc0_std, color='blue', linestyle='dotted', linewidth=1)

        ax2.axhline(cost0_avg, color='red', linestyle='-', linewidth=1.5)
        ax2.axhline(cost0_avg + cost0_std, color='red', linestyle='dotted', linewidth=1)
        ax2.axhline(cost0_avg - cost0_std, color='red', linestyle='dotted', linewidth=1)

        # X-axis labels
        ax1.set_xticks(ind)
        ax1.set_xticklabels(algorithms, rotation=90)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, bottom=0.3)  # Increased bottom margin
    plt.savefig(filename)
    plt.close()



data = '''HetFoA
13/25
0.475
14/25
0.478
14/25
0.476
14/25
1.408
15/25
1.407
16/25
1.4608
15/25
0.729
14/25
0.6123
13/25
0.683

Dynamic Composition
18/25
0.408
15/25
0.417
16/25
0.437
15/25
1.36
16/25
1.301
15/25
1.476
16/25
0.738
16/25
0.709
13/25
0.7966

Difficulty Based Width Initialization
16/25
0.338
11/25
0.432
16/25
0.3697
14/25
1.154
15/25
1.392
16/25
1.05
13/25
0.644
15/25
0.545
16/25
0.5274

Runtime Width Adaptation
17/25
0.544
18/25
0.569
15/25
0.609
15/25
1.558
14/25
1.576
14/25
1.42
15/25
0.87
15/25
0.823
17/25
0.789

Runtime Width Adaptation + Difficulty Based Width Initialization
16/25
0.4346
14/25
0.5034
17/25
0.5157
17/25
1.211
17/25
1.464
16/25
1.533
14/25
0.591
12/25
0.6104
17/25
0.667

Difficulty Based Width Initialization + Dynamic composition
16/25
0.3794
18/25
0.388
20/25
0.348
11/25
1.364
15/25
1.155
13/25
1.39
15/25
0.498
16/25
0.5397
14/25
0.601

Skewed State Detection (V1)
14/25
0.5367
17/25
0.5329
17/25
0.4954
16/25
1.568
17/25
1.527
15/25
1.555
17/25
0.795
16/25
0.786
14/25
0.6354

Runtime Width Adaptation + Skewed State Detection (V1)
17/25
0.4921
15/25
0.4868
17/25
0.4897
17/25
1.113
14/25
1.4664
14/25
1.566
15/25
0.707
16/25
0.7136
14/25
0.734
'''




tasks = data.split('\n\n')
env_data = {
    'game24': [],
    'hotpotqa': [],
    'scibench': []
}
env_idxs = ['game24', 'hotpotqa', 'scibench']
algos = []

for task in tasks:
    lines = task.splitlines()
    algo = lines[0]
    algos.append(algo)
    nums = list(map(eval, lines[1:]))
    
    for start in range(0, 6*len(env_idxs), 6):
        env = env_idxs[start//6]
        acc = []
        cost = []
        for i in range(start, start+6):
            if i % 2 == 0: # acc
                acc.append(nums[i])
            else:
                cost.append(nums[i])

        acc_avg = sum(acc)/len(acc)
        acc_var = sum([(num - acc_avg)**2 for num in acc]) / len(acc)
        acc_stdev = (acc_var)**0.5

        cost_avg = sum(cost)/len(cost)
        cost_var = sum([(num - cost_avg)**2 for num in cost]) / len(cost)
        cost_stdev = (cost_var)**0.5

        env_data[env].append((acc_avg, acc_stdev, cost_avg, cost_stdev))


plot_algo_bars(algos, env_data)