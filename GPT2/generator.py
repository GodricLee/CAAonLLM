import torch
import random
import numpy as np
from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

def truncate_repetition_10(tokens, k=10):
    seen = set()
    for i in range(len(tokens) - k + 1):
        snippet = tuple(tokens[i:i + k])
        if snippet in seen:
            return tokens[:i]
        seen.add(snippet)
    return tokens

def text_generator(state_dict, args):
    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print(args.text)
    context_tokens = enc.encode(args.text)

    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens if not args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

        out = out[:, len(context_tokens):].tolist()
        for i in range(args.batch_size):
            # print("iii:", i)
            generated += 1
            post_out = out[i]
            if enc.encoder['<|endoftext|>'] in post_out:
                post_out = post_out[:post_out.index(enc.encoder['<|endoftext|>'])]
            # 从前往后截断重复
            # print(enc.decode(post_out))
            post_out = truncate_repetition_10(post_out, 10)
            text = enc.decode(post_out)
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)
