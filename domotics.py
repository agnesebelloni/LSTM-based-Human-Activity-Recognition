
import numpy as np
X_test = np.loadtxt("/Users/computer/Desktop/project domotics/UCI HAR Dataset/test/X_test.txt")
y_test = np.loadtxt("/Users/computer/Desktop/project domotics/UCI HAR Dataset/test/y_test.txt")

X_train = np.loadtxt("/Users/computer/Desktop/project domotics/UCI HAR Dataset/train/X_train.txt")
y_train = np.loadtxt("/Users/computer/Desktop/project domotics/UCI HAR Dataset/train/y_train.txt")

activity_map ={}
for line in open("/Users/computer/Desktop/project domotics/UCI HAR Dataset/activity_labels.txt", "r"):
    idx, name =line.strip().split()
    activity_map[int(idx)] = name

print("Train:", X_train.shape, y_train.shape)
print("Test: ", X_test.shape,  y_test.shape)
print("Activity map:", activity_map)

#faccio un conteggio sul train per vedere se le classi delle attività sono bilanciate o meno
activity_counts_train = {}

for label in y_train:
    name = activity_map[label]
    activity_counts_train[name] = activity_counts_train.get(name, 0) + 1 

print(activity_counts_train) #qui poi posssiamo fare un istogramma ma poi vediamo
#comunque le classi sono abbastanza bilanciate quindi nessuna è sottorappresentata
activity_counts_test = {}
for label in y_test:
    name = activity_map[label]
    activity_counts_test[name] = activity_counts_test.get(name, 0) + 1 

print(activity_counts_test) 

#media delle classi
class_means = {}

for label, name in activity_map.items():
    X_class = X_train[y_train == label]
    class_means[name] = X_class.mean(axis=0)

print(class_means)
print(class_means.keys)

from itertools import combinations
from numpy.linalg import norm

print("\nPairwise distances between activities:\n")

for (name1, mu1), (name2, mu2) in combinations(class_means.items(), 2):
    dist = norm(mu1 - mu2)
    print(f"{name1:20s} vs {name2:20s} -> distance = {dist:.2f}")

#suddivisione tasks

task1 = [1, 2, 3]
task2 = [4, 5]
task3 = [6]

TASKS = [task1, task2, task3]

#creo i dataset training suddivisi per task
trainsets = []

for task_labels in TASKS:
    X_task = []
    y_task = []

    for i in range(len(y_train)):
        if y_train[i] in task_labels:
            X_task.append(X_train[i])
            y_task.append(y_train[i])

    X_task = np.array(X_task)
    y_task = np.array(y_task)

    trainsets.append((X_task, y_task))

print(trainsets)

seen_testsets = []

seen_labels = set()

for task_labels in TASKS:
   
    seen_labels.update(task_labels)

    X_seen = []
    y_seen = []

    for i in range(len(y_test)):
        if y_test[i] in seen_labels:
            X_seen.append(X_test[i])
            y_seen.append(y_test[i])

    X_seen = np.array(X_seen)
    y_seen = np.array(y_seen)

    seen_testsets.append((X_seen, y_seen))

    #baseline "offline" (upper bound)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

offline_model = LogisticRegression(max_iter=2000, n_jobs=-1)
offline_model.fit(X_train, y_train)

y_pred_off = offline_model.predict(X_test)
print("OFFLINE accuracy:", accuracy_score(y_test, y_pred_off))

#incremental setting "naive fine tuning"
inc_model = LogisticRegression(max_iter=2000, n_jobs=-1, warm_start=True)

acc_after_each_task = []

for t_idx, (Xt, yt) in enumerate(trainsets):
    
    pass

#icremental vero con partial_fit
from sklearn.linear_model import SGDClassifier

classes_all = np.array(sorted(activity_map.keys()))  # [1..6]

inc_model = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", tol=None)

acc_after_each_task = []

for task_idx, (Xt, yt) in enumerate(trainsets):
    # training incrementale sul task corrente
    inc_model.partial_fit(Xt, yt, classes=classes_all)

    # evaluation "seen-so-far"
    Xs, ys = seen_testsets[task_idx]
    y_pred = inc_model.predict(Xs)

    acc = accuracy_score(ys, y_pred)
    acc_after_each_task.append(acc)

    print(f"After Task {task_idx+1} | seen labels = {np.unique(ys)} | acc = {acc:.4f}")

#grafico andamento accuracy
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, len(acc_after_each_task)+1), acc_after_each_task, marker="o")
plt.xlabel("Task")
plt.ylabel("Seen-so-far Accuracy")
plt.title("Incremental learning (naive) - Seen-so-far accuracy")
plt.xticks(range(1, len(acc_after_each_task)+1))
plt.show()

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# Hyperparametri replay
# ---------------------------
MEMORY_PER_CLASS = 50   # esempi salvati per ogni classe (puoi provare 10, 20, 50, 100)

# classi globali (1..6)
classes_all = np.array(sorted(activity_map.keys()))

# modello incrementale con partial_fit
replay_model = SGDClassifier(
    loss="log_loss",
    max_iter=1,
    learning_rate="optimal",
    tol=None,
    random_state=0
)

# ---------------------------
# Memory buffer: dizionario {label: (X_mem_label, y_mem_label)}
# ---------------------------
memory = {}  # label -> array di esempi

def update_memory_uniform_per_class(memory, X_new, y_new, memory_per_class):
    """
    Aggiorna la memoria scegliendo fino a 'memory_per_class' campioni per ogni classe
    tra quelli disponibili in X_new,y_new (tipicamente dal task corrente).
    """
    for c in np.unique(y_new):
        Xc = X_new[y_new == c]

        # se ci sono più campioni di quelli che vuoi, campiona random
        if len(Xc) > memory_per_class:
            idx = np.random.choice(len(Xc), size=memory_per_class, replace=False)
            Xc = Xc[idx]

        memory[c] = Xc  # salva solo X; y lo ricrei con c
    return memory

def build_replay_batch(memory):
    """
    Converte il dizionario memory in (X_mem, y_mem) concatenati.
    """
    if len(memory) == 0:
        return None, None

    X_list = []
    y_list = []
    for c, Xc in memory.items():
        X_list.append(Xc)
        y_list.append(np.full(len(Xc), c))

    X_mem = np.vstack(X_list)
    y_mem = np.concatenate(y_list)
    return X_mem, y_mem

# ---------------------------
# Training incrementale con replay
# ---------------------------
acc_replay = []

for task_idx, (Xt, yt) in enumerate(trainsets):

    # 1) costruisci batch di replay (memoria)
    X_mem, y_mem = build_replay_batch(memory)

    # 2) dataset effettivo di training = Xt + memoria
    if X_mem is None:
        X_train_eff = Xt
        y_train_eff = yt
    else:
        X_train_eff = np.vstack([Xt, X_mem])
        y_train_eff = np.concatenate([yt, y_mem])

    # 3) update incrementale
    replay_model.partial_fit(X_train_eff, y_train_eff, classes=classes_all)

    # 4) evaluation "seen-so-far"
    Xs, ys = seen_testsets[task_idx]
    y_pred = replay_model.predict(Xs)
    acc = accuracy_score(ys, y_pred)
    acc_replay.append(acc)

    print(f"[REPLAY] After Task {task_idx+1} | seen labels = {np.unique(ys)} | acc = {acc:.4f}")

    # 5) aggiorna la memoria usando i dati del task corrente
    memory = update_memory_uniform_per_class(memory, Xt, yt, MEMORY_PER_CLASS)

import numpy as np
from sklearn.metrics import accuracy_score

def per_class_accuracy(y_true, y_pred, labels):
    """
    Ritorna dict: {label: acc_label} calcolata solo sui campioni di quella label.
    """
    out = {}
    for c in labels:
        mask = (y_true == c)
        if mask.sum() == 0:
            out[c] = np.nan
        else:
            out[c] = accuracy_score(y_true[mask], y_pred[mask])
    return out

# ---- NAIVE: devi salvare le predizioni dopo ogni task (se non l'hai fatto, rifai loop)
# Supponiamo tu abbia inc_model (naive) e replay_model (replay) già allenati step-by-step:

import numpy as np
from sklearn.metrics import accuracy_score

def per_class_accuracy(y_true, y_pred, labels):
    """
    Ritorna dict: {label: acc_label} calcolata solo sui campioni di quella label.
    """
    out = {}
    for c in labels:
        mask = (y_true == c)
        if mask.sum() == 0:
            out[c] = np.nan
        else:
            out[c] = accuracy_score(y_true[mask], y_pred[mask])
    return out

# ---- NAIVE: devi salvare le predizioni dopo ogni task (se non l'hai fatto, rifai loop)
# Supponiamo tu abbia inc_model (naive) e replay_model (replay) già allenati step-by-step:

def compute_forgetting(perclass_history, tasks, classes_all):
    """
    perclass_history: lista lunga = num_tasks, ogni elemento è dict {label: acc} per labels viste fino lì
    tasks: lista di task labels (es. [[1,2,3],[4,5],[6]])
    ritorna dict: {label: forgetting}
    """
    # task di introduzione per ogni classe
    intro_task = {}
    for t_idx, t in enumerate(tasks):
        for c in t:
            intro_task[c] = t_idx

    # accuracy finale per classe (all'ultimo task)
    final_dict = perclass_history[-1]

    forgetting = {}
    for c in classes_all:
        t0 = intro_task[c]  # da quando esiste
        # best acc dalla sua introduzione in poi
        best = np.nan
        for k in range(t0, len(perclass_history)):
            if c in perclass_history[k]:
                val = perclass_history[k][c]
                if not np.isnan(val):
                    best = val if np.isnan(best) else max(best, val)

        final = final_dict.get(c, np.nan)

        forgetting[c] = best - final if (not np.isnan(best) and not np.isnan(final)) else np.nan

    return forgetting

naive_forgetting = compute_forgetting(naive_perclass_history, TASKS, classes_all)
replay_forgetting = compute_forgetting(replay_perclass_history, TASKS, classes_all)

# stampa leggibile con nomi
print("\nForgetting per class (NAIVE):")
for c in classes_all:
    print(f"{activity_map[c]:20s} -> {naive_forgetting[c]:.4f}")

print("\nForgetting per class (REPLAY):")
for c in classes_all:
    print(f"{activity_map[c]:20s} -> {replay_forgetting[c]:.4f}")

import matplotlib.pyplot as plt
import numpy as np

labels = [activity_map[c] for c in classes_all]
x = np.arange(len(classes_all))

naive_vals = [naive_forgetting[c] for c in classes_all]
replay_vals = [replay_forgetting[c] for c in classes_all]

plt.figure()
plt.bar(x - 0.2, naive_vals, width=0.4, label="Naive")
plt.bar(x + 0.2, replay_vals, width=0.4, label="Replay")
plt.xticks(x, labels, rotation=30, ha="right")
plt.ylabel("Forgetting (best acc - final acc)")
plt.title("Forgetting per class: Naive vs Replay")
plt.legend()
plt.tight_layout()
plt.show()
