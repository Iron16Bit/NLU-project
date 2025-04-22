import os
from utils import lang, test_loader, train_loader, dev_loader
from config import PAD_TOKEN, device, clip, emb_size, hid_size, lr, n_epochs, runs, dropouts
from model import ModelIAS, ModelIAS_bi, ModelIAS_drop
from functions import init_weights, train_loop, eval_loop
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Global variables
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)

criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss()

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)

# #! STANDARD TRAINING
# # Store information about all runs
# all_losses_train = []
# all_losses_dev = []
# all_sampled_epochs = []

# slot_f1s, intent_acc = [], []

# # Train
# for lr in lrs:
#     for run in tqdm(range(0, runs)):
#         model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
#         model.apply(init_weights)

#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
#         criterion_intents = nn.CrossEntropyLoss()

#         patience = 3
#         losses_train = []
#         losses_dev = []
#         sampled_epochs = []
#         best_f1 = 0

#         for epoch in range(1, n_epochs):
#             loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
            
#             if epoch % 5 == 0:
#                 sampled_epochs.append(epoch)
#                 losses_train.append(np.asarray(loss).mean())

#                 results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
#                 losses_dev.append(np.asarray(loss_dev).mean())

#                 f1 = results_dev['total']['f']
#                 if f1 > best_f1:
#                     best_f1 = f1
#                 else:
#                     patience -= 1
#                 if patience <= 0:
#                     break

#         all_losses_train.append(losses_train)
#         all_losses_dev.append(losses_dev)
#         all_sampled_epochs.append(sampled_epochs)

#         results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
#         intent_acc.append(intent_test['accuracy'])
#         slot_f1s.append(results_test['total']['f'])

# # Print mean F1 score and intent accuracy across all runs
# slot_f1s = np.asarray(slot_f1s)
# intent_acc = np.asarray(intent_acc)
# print('Slot F1', round(slot_f1s.mean(), 3), '+-', round(slot_f1s.std(), 3))
# print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))

# -----------------------------------------------------------------------------------------------------------------------------

# #? TRAIN OVER DIFFERENT LERANING RATES
# # Store information about all runs
# all_losses_train = []
# all_losses_dev = []
# all_sampled_epochs = []

# slot_f1s_all = []
# intent_acc_all = []

# # Train
# for lr in lrs:
#     print(f"\nTraining with learning rate: {lr}")
    
#     slot_f1s = []
#     intent_acc = []

#     for run in tqdm(range(0, runs)):
#         model = ModelIAS_bi(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
#         model.apply(init_weights)

#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
#         criterion_intents = nn.CrossEntropyLoss()

#         patience = 3
#         losses_train = []
#         losses_dev = []
#         sampled_epochs = []
#         best_f1 = 0

#         for epoch in range(1, n_epochs):
#             loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)

#             if epoch % 5 == 0:
#                 sampled_epochs.append(epoch)
#                 losses_train.append(np.asarray(loss).mean())

#                 results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
#                 losses_dev.append(np.asarray(loss_dev).mean())

#                 f1 = results_dev['total']['f']
#                 if f1 > best_f1:
#                     best_f1 = f1
#                 else:
#                     patience -= 1
#                 if patience <= 0:
#                     break

#         all_losses_train.append(losses_train)
#         all_losses_dev.append(losses_dev)
#         all_sampled_epochs.append(sampled_epochs)

#         results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
#         intent_acc.append(intent_test['accuracy'])
#         slot_f1s.append(results_test['total']['f'])

#     # After all runs for this lr
#     slot_f1s = np.asarray(slot_f1s)
#     intent_acc = np.asarray(intent_acc)
#     slot_f1s_all.append(slot_f1s)
#     intent_acc_all.append(intent_acc)

#     print(f'LR {lr} -> Slot F1: {round(slot_f1s.mean(), 3)} ± {round(slot_f1s.std(), 3)}')
#     print(f'LR {lr} -> Intent Acc: {round(intent_acc.mean(), 3)} ± {round(intent_acc.std(), 3)}')

# print("\nSummary of all learning rates:")
# for i, lr in enumerate(lrs):
#     print(f'LR {lr} -> Slot F1: {round(slot_f1s_all[i].mean(), 3)} ± {round(slot_f1s_all[i].std(), 3)}, '
#           f'Intent Acc: {round(intent_acc_all[i].mean(), 3)} ± {round(intent_acc_all[i].std(), 3)}')

# -------------------------------------------------------------------------------------------------------------

#! ITERATE OVER DIFFERENT DROPOUTS
# Store information about all runs
all_losses_train = []
all_losses_dev = []
all_sampled_epochs = []

slot_f1s_all = []
intent_acc_all = []

for dropout_rate in dropouts:
    print(f"\n=== Training with Dropout: {dropout_rate} ===")

    slot_f1s = []
    intent_acc = []

    for run in tqdm(range(0, runs)):
        model = ModelIAS_drop(
            hid_size, out_slot, out_int, emb_size, vocab_len,
            pad_index=PAD_TOKEN, dropout=dropout_rate
        ).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0

        for epoch in range(1, n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)

            if epoch % 5 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())

                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())

                f1 = results_dev['total']['f']
                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0:
                    break

        all_losses_train.append(losses_train)
        all_losses_dev.append(losses_dev)
        all_sampled_epochs.append(sampled_epochs)

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

    # After all runs for this dropout value
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    slot_f1s_all.append(slot_f1s)
    intent_acc_all.append(intent_acc)

    print(f'Dropout {dropout_rate} -> Slot F1: {round(slot_f1s.mean(), 3)} ± {round(slot_f1s.std(), 3)}')
    print(f'Dropout {dropout_rate} -> Intent Acc: {round(intent_acc.mean(), 3)} ± {round(intent_acc.std(), 3)}')

print("\n=== Summary of All Dropout Settings ===")
for i, dropout_rate in enumerate(dropouts):
    print(f'Dropout {dropout_rate} -> Slot F1: {round(slot_f1s_all[i].mean(), 3)} ± {round(slot_f1s_all[i].std(), 3)}, '
          f'Intent Acc: {round(intent_acc_all[i].mean(), 3)} ± {round(intent_acc_all[i].std(), 3)}')

# Find the common set of sampled epochs (min length)
min_len = min(len(x) for x in all_sampled_epochs)
trimmed_epochs = all_sampled_epochs[0][:min_len]  # Use epoch points from first run (trimmed)

# Trim all to same length and compute mean across runs
all_losses_train = np.array([run[:min_len] for run in all_losses_train])
all_losses_dev = np.array([run[:min_len] for run in all_losses_dev])

mean_train_loss = all_losses_train.mean(axis=0)
mean_dev_loss = all_losses_dev.mean(axis=0)

# Plot of the train and valid losses
plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor('white')
plt.title('Mean Train and Dev Losses over Runs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(trimmed_epochs, mean_train_loss, label='Train loss')
plt.plot(trimmed_epochs, mean_dev_loss, label='Dev loss')
plt.legend()
plt.grid(True)
plt.show()