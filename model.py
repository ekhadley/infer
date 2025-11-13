import torch as t

import transformer_lens
from transformer_lens import HookedTransformer

def load_model(model_id: str, device: str = "cuda", dtype: str = "float32") -> HookedTransformer:
    model = HookedTransformer.from_pretrained_no_processing(
        model_name = model_id,
        device = t.device(device),
        dtype = dtype,
    )
    return model

class Model:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float32"):
        self.model = load_model(model_id, device, dtype)
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.cfg = self.model.cfg
        self.tokenizer = self.model.tokenizer

    @t.inference_mode()
    def get_distn(self, prompt: str, topk: int = 10) -> dict:
        distn = {"tok_ids": [], "tok_strs": [], "logits": [], "probs": []}

        print(prompt)
        toks = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device).squeeze()
        print(toks)
        logits = self.model(toks)
        print(logits.shape)
        last_logits = logits[0, -1]
        print(last_logits.shape)
        probs = t.softmax(last_logits, dim=-1)

        top_logits = t.topk(probs, k=topk)
        for i in range(topk):
            distn["tok_ids"].append(top_logits.indices[i].item())
            distn["tok_strs"].append(self.tokenizer.decode(top_logits.indices[i].item()))
            distn["logits"].append(top_logits.values[i].item())
            distn["probs"].append(probs[top_logits.indices[i].item()].item())
    
        return distn

    @t.inference_mode()
    def get_all_positions_distn(self, prompt: str, topk: int = 10) -> dict:
        """
        Get distributions for all token positions in the prompt.
        Returns: {
            "tokens": [list of token strings],
            "token_ids": [list of token ids],
            "distributions": [list of distributions, one per position]
        }
        """
        # Tokenize the prompt
        toks = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device).squeeze()
        
        # Get token strings
        token_strs = [self.tokenizer.decode(tok.item()) for tok in toks]
        token_ids = [tok.item() for tok in toks]
        
        # Get distributions for each position
        distributions = []
        for i in range(len(toks)):
            # Get logits for prefix [0:i+1]
            prefix_toks = toks[:i+1].unsqueeze(0)
            logits = self.model(prefix_toks)
            last_logits = logits[0, -1]
            probs = t.softmax(last_logits, dim=-1)
            
            # Get top-k
            top_probs = t.topk(probs, k=topk)
            
            pos_distn = {
                "tok_ids": [top_probs.indices[j].item() for j in range(topk)],
                "tok_strs": [self.tokenizer.decode(top_probs.indices[j].item()) for j in range(topk)],
                "probs": [top_probs.values[j].item() for j in range(topk)]
            }
            distributions.append(pos_distn)
        
        return {
            "tokens": token_strs,
            "token_ids": token_ids,
            "distributions": distributions
        }

import tabulate
def display_distn(distn: dict):
    table = []
    for i in range(len(distn["tok_ids"])):
        table.append([distn["tok_ids"][i], distn["tok_strs"][i], distn["logits"][i], distn["probs"][i]])
    print(tabulate.tabulate(table, headers=["Tok ID", "Tok Str", "Logit", "Prob"]))
        
if __name__ == "__main__":
    model = Model(model_id = "gpt2-small", device="cuda")

    prompt = "Hello, how are"
    distn = model.get_distn(prompt)
    display_distn(distn)
