import torch
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

def seed_everything(seed=1062):
  np.random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def test_model(solution, Xtrain, X_test, ytrain, y_test):
    seed_everything(1006)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    List_acc_kfold, List_valid_kfold, List_test_kfold = [], [], []

    for train_ind, test_ind in kfold.split(Xtrain, ytrain):
        X_train, X_valid = Xtrain[train_ind], Xtrain[test_ind]
        y_train, y_valid = ytrain[train_ind], ytrain[test_ind]
        model = TabNetClassifier(n_d=solution.N_d, n_a=solution.N_a, n_steps=solution.N_steps, gamma=1.5, n_independent=2, n_shared=2, cat_emb_dim=1, lambda_sparse=1e-4, momentum=0.3, clip_value=2., optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=solution.l_r), scheduler_params = {"gamma": 0.95, "step_size": 20}, scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15)
        model.fit(X_train=X_train, y_train=y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_name=['train', 'valid'], max_epochs=90, patience=100, batch_size=1024, virtual_batch_size=256) # batch_size=1024, virtual_batch_size=256
        List_acc_kfold.append(model.history['train_accuracy'][model.best_epoch])
        List_valid_kfold.append(model.best_cost)
        preds_mapper = {idx : class_name for idx, class_name in enumerate(model.classes_)}
        List_test_kfold.append(accuracy_score(y_pred=np.vectorize(preds_mapper.get)(np.argmax(model.predict_proba(X_test), axis=1)), y_true=y_test))
    return List_acc_kfold, List_valid_kfold, List_test_kfold # Return train accuracy, error valid and error test

def fitness_function(solution, Xtrain, ytrain):
    seed_everything(1006)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results_valid = []
    for train_ind, test_ind in kfold.split(Xtrain, ytrain):
        X_train, X_valid = Xtrain[train_ind], Xtrain[test_ind]
        y_train, y_valid = ytrain[train_ind], ytrain[test_ind]
        model = TabNetClassifier(n_d=solution.N_d, n_a=solution.N_a, n_steps=solution.N_steps, gamma=1.5, n_independent=2, n_shared=2, cat_emb_dim=1, lambda_sparse=1e-4, momentum=0.3, clip_value=2., optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=solution.l_r), scheduler_params = {"gamma": 0.95, "step_size": 20}, scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15)
        model.fit(X_train=X_train, y_train=y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_name=['train', 'valid'], max_epochs=90, patience=100, batch_size=1024, virtual_batch_size=256)  # batch_size=1024, virtual_batch_size=256
        results_valid.append(100.0*(1-model.best_cost))
    print("N_d: {}, N_a: {}, L_r: {}, N_steps: {}, error_valid: {:.6f}\n".format(solution.N_d, solution.N_a, solution.l_r, solution.N_steps, np.mean(results_valid)))
    return np.mean(results_valid), np.std(results_valid) # Return error valid pos-train

def greedy_function(solution, Xtrain, ytrain):
    seed_everything(1006)
    results_valid = [] # For fold results
    X_train, X_valid, y_train, y_valid = train_test_split(Xtrain, ytrain, test_size=.10, random_state=1, shuffle=True)
    model = TabNetClassifier(n_d=solution.N_d, n_a=solution.N_a, n_steps=solution.N_steps, gamma=1.5, n_independent=2, n_shared=2, cat_emb_dim=1, lambda_sparse=1e-4, momentum=0.3, clip_value=2., optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=solution.l_r), scheduler_params = {"gamma": 0.95, "step_size": 20}, scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15)
    model.fit(X_train=X_train, y_train=y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_name=['train', 'valid'], max_epochs=90, patience=100, batch_size=1024, virtual_batch_size=256) # batch_size=1024, virtual_batch_size=256
    results_valid.append(100.0*(1-model.best_cost))
    print("Error accuracy valid:", np.mean(results_valid))
    return np.mean(results_valid)
