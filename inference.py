import torch

def inference(sentence, model, encoder, device):
    model.eval()
    idx = torch.tensor(encoder.encode(sentence), dtype=torch.long).unsqueeze(0)
    idx = idx.to(device)
    generated = model.generate(idx, max_new_tokens=100)
    print(generated)
    res = encoder.decode(generated[0].cpu().numpy())
    return res