import torch
from ..countdown.countdown_task import answer_reward_function as reward_function

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    with torch.inference_mode():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs

    # Compute rewards for batch texts
    rewards = []
    for i, (gen_text, tgt_text, inp_text) in enumerate(zip(generated_texts, target_texts, input_texts)):
        numbers = None
        target = None
        if "[" in inp_text and "]" in inp_text:
            start_idx = inp_text.find("[")
            end_idx = inp_text.find("]")
            if start_idx != -1 and end_idx != -1:
                numbers_str = inp_text[start_idx+1:end_idx]
                numbers = [int(n) for n in numbers_str.split() if n.isdigit()]

        if tgt_text.isdigit():
            target = int(tgt_text)

        model_response = gen_text
        if "assistant:" in gen_text:
            model_response = gen_text.split("assistant:")[-1].strip()

        # Use reward_function from countdown_task.py
        reward_result = reward_function(model_response, numbers, target)
        rewards.append(reward_result)


    if return_text:
        return rewards, generated_texts
    else:
        return rewards