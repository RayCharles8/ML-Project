import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Definite values
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Functions for training and evaluation
def train_gnn_model(model, train_loader, val_loader, loss_fn, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_acc, val_f1, val_auc = [], [], [], []

    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate on validation set
        acc,f1,auc = evaluate_gnn_model(model, val_loader)

        # Store metrics
        train_loss.append(avg_loss)
        val_acc.append(acc)
        val_f1.append(f1)
        val_auc.append(auc)

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    model_metrics = {
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_auc": val_auc
    }

    return model_metrics

def evaluate_gnn_model(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

    model.eval()

    eval_true, eval_pred, eval_probs = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Prediction
            pred = model(data)
            true = data.y
            prob = torch.softmax(pred, dim=1)[:,1]

            eval_true.extend(true.cpu().numpy())
            eval_pred.extend(pred.argmax(dim=1).cpu().numpy())
            eval_probs.extend(prob.cpu().numpy())

    eval_acc = accuracy_score(eval_true, eval_pred)
    f1 = f1_score(eval_true, eval_pred, zero_division=0)
    auc = roc_auc_score(eval_true, eval_probs)

    scheduler.step(eval_acc)
    return eval_acc, f1, auc


def evaluate_tabular_model(model, X_data, y_true, name="Set"):
    y_pred = model.predict(X_data)
    y_prob = model.predict_proba(X_data)[:,1]

    print(f"\n{name} Performance")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))
    print(classification_report(y_true, y_pred))