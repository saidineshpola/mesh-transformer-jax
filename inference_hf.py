import time
import torch
from transformers import GPTJForCausalLM, GPT2Tokenizer
import transformers
import argparse
from transformers import AutoTokenizer
if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    if torch.cuda.is_available():
        model =  GPTJForCausalLM.from_pretrained("./gpt-j-6B", torch_dtype=torch.float16).cuda()
    else:
        model =  GPTJForCausalLM.from_pretrained("./gpt-j-6B")

    # Initialize parser
    my_parser = argparse.ArgumentParser(description='Give the input text to GPT-J')
    my_parser.add_argument('--input', metavar='INPUT',
                        type=str,
                        help='input text')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    input_text = args.input
    input_ids = tokenizer(str(input_text), return_tensors='pt').input_ids

#     output = model.generate(
#         input_ids,
#         do_sample=True,
#         max_length=20,
#         top_p=0.7,
#         top_k=0,
#         temperature=1.0,
#     )    
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    print("--"*50)
    print(tokenizer.batch_decode(gen_tokens)[0])
