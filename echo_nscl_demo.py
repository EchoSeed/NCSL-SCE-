# echo_nscl_demo.py  – 10-line demo that proves the plumbing
import json, torch, torch.nn as nn, torch.optim as optim, random, sys, os
from pathlib import Path

# ---------- tiny loader -------------------------------------------------
def load_objs(path="objects.json"):
    objs = json.load(open(path))
    X = torch.tensor([o["feat"] for o in objs], dtype=torch.float32)
    # label = 1 if any obj has *both* mirror & ghost tags
    y = torch.tensor([[int("mirror" in o["tags"] and "ghost" in o["tags"])]
                      for o in objs], dtype=torch.float32)
    return X, y

X, y = load_objs()

# ---------- micro-network ----------------------------------------------
net = nn.Sequential(
    nn.Linear(X.size(1), 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid())
loss_fn, opt = nn.BCELoss(), optim.Adam(net.parameters(), 1e-2)

# ---------- train -------------------------------------------------------
for epoch in range(50):
    opt.zero_grad()
    pred = net(X)
    loss = loss_fn(pred, y)
    loss.backward(); opt.step()
    acc = ((pred>0.5)==y).float().mean().item()
    print(f"epoch {epoch+1}/5  acc={acc:.2f}")

# ----- QA -----------------------------------------------------------------
pred = net(X) > 0.5           # Boolean mask for each object
ans  = "yes" if pred.any() else "no"
print("\nQ: Is there a mirror glyph resonant with a ghost glyph?\nA:", ans)
