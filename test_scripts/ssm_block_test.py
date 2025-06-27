import torch
from tqdm import tqdm

from core.model.ssm.ssm_block import SSM_Block

torch.cuda.memory._record_memory_history(
    max_entries=100000
)

bs, w, h = 1, 28, 28
seq_len, channels = 320, 256
device = "cuda"

block = SSM_Block(seq_len=seq_len, hidden_states=64, in_shape=(channels, w, h))

block.to(device)
#block.compile()

# dummy training sim
num_epochs, num_data = 1, 4
optimizer = torch.optim.AdamW(block.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = torch.amp.GradScaler('cuda')
for _ in range(num_epochs):
    for _ in tqdm(range(num_data)):
        u_t = torch.randn(bs * seq_len, channels, w, h, device=device) * 10
        t_t = torch.randn(bs * seq_len, channels, w, h, device=device) * 10
        with torch.amp.autocast('cuda'):
            outputs = block(u_t)
            loss = torch.dist(outputs, t_t)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    scheduler.step()

block.eval()

bs = 1
u = torch.randn(bs * seq_len, channels, w, h, device=device)

with torch.no_grad():
    print("-------------------------------------------------")
    res1 = block(u)

    block.check_stability()
    block.set_mode(to_rnn=True, device=device)

    print("-------------------------------------------------")

    for i in range(seq_len):
        # не учитывается размер батча
        _u = u[i, :, :, :].unsqueeze(0)
        res2 = block(_u)
        # print(res2)

    print("-------------------------------------------------")

    print(torch.mean(torch.abs(res1[-1, :, :, :] - res2)))


try:
   torch.cuda.memory._dump_snapshot(f"mem.pickle")
except Exception as e:
   print(f"Failed to capture memory snapshot {e}")

# Stop recording memory snapshot history.
torch.cuda.memory._record_memory_history(enabled=None)