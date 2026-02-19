import torch

def format_chat_for_inference(question: str) -> str:
    """格式化问题用于推理"""
    system_prompt = "You are a biomedical assistant. Answer the question with the most appropriate entity name."
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def get_prediction_metrics(logits, target_id, metric_type="logit"):
    """
    Calculate metrics for target token(s) at the last position(s).
    Supports both single token and multi-token targets.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        target_id: Single token ID (int) or list of token IDs for multi-token answers
        metric_type: Type of metric to calculate - "logit" or "mrr"
        
    Returns:
        metric_value: The average metric value across all target tokens
    """
    # Handle both single token and multiple tokens
    if isinstance(target_id, int):
        target_ids = [target_id]
    else:
        target_ids = list(target_id)
    
    n_tokens = len(target_ids)
    
    if metric_type == "logit":
        # Calculate logit value for each target token at corresponding position
        logit_values = []
        for i, tid in enumerate(target_ids):
            pos = -(n_tokens - i)  # Position from the end
            # pos左移一位，因为加上answer后预测的是下一个token
            pos -= 1
            logit_value = logits[0, pos, tid].item()
            logit_values.append(logit_value)
        
        # Return average logit value
        return sum(logit_values) / len(logit_values)
    
    elif metric_type == "mrr":
        # Calculate MRR for each target token at corresponding position
        mrr_values = []
        for i, tid in enumerate(target_ids):
            pos = -(n_tokens - i)  # Position from the end
            # pos左移一位，因为加上answer后预测的是下一个token
            pos -= 1
            position_logits = logits[0, pos, :]
            
            # Calculate rank (1-indexed: rank=1 means highest logit)
            sorted_indices = torch.argsort(position_logits, descending=True)
            rank = (sorted_indices == tid).nonzero(as_tuple=True)[0].item() + 1
            # Calculate MRR for this position
            mrr = 1.0 / rank
            mrr_values.append(mrr)
        # Return average MRR
        return sum(mrr_values) / len(mrr_values)
    
    else:
        raise ValueError(f"Invalid metric_type: {metric_type}. Must be 'logit' or 'mrr'")
    


def patch_residual_stream(hooked_model, layer, positions, clean_cache, corrupted_tokens, patch_type='residual'):
    """
    Patch the residual stream, attention output, or MLP output at a specific layer and position(s).
    
    Args:
        hooked_model: The hooked model to run
        layer: Layer index to patch
        positions: Single position (int) or list of positions to patch simultaneously
        clean_cache: Cache of clean activations
        corrupted_tokens: Corrupted input tokens
        patch_type: Type of patching - 'residual', 'attention', or 'mlp'
    
    Returns:
        patched_logits: Model output after patching
    """
    if patch_type == 'residual':
        hook_name = f"blocks.{layer}.hook_resid_pre"
    elif patch_type == 'attention':
        hook_name = f"blocks.{layer}.hook_attn_out"
    elif patch_type == 'mlp':
        hook_name = f"blocks.{layer}.hook_mlp_out"
    else:
        raise ValueError(f"Invalid patch_type: {patch_type}. Must be 'residual', 'attention', or 'mlp'")
    
    clean_activation = clean_cache[hook_name]
    
    # Convert single position to list for uniform handling
    if isinstance(positions, int):
        positions = [positions]
    
    def hook_fn(act, hook):
        # Patch all specified positions
        for pos in positions:
            act[:, pos, :] = clean_activation[:, pos, :]
        return act
    
    patched_logits = hooked_model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, hook_fn)]
    )
    
    return patched_logits


# noise based corruption

# ROME-style implementation: Add Gaussian noise to embeddings

def get_noised_embeddings(hooked_model, tokens, positions, noise_std=3.0, seed=None):
    """
    Get embeddings with Gaussian noise added at specified positions.
    
    Args:
        tokens: Input tokens [batch, seq_len]
        positions: Single position (int) or list of positions to add noise to
        noise_std: Standard deviation of Gaussian noise (default: 3.0, as in ROME paper)
        seed: Random seed for reproducibility
        
    Returns:
        noised_embeddings: Embeddings with noise added [batch, seq_len, d_model]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Convert single position to list for uniform handling
    if isinstance(positions, int):
        positions = [positions]
    
    # Get clean embeddings
    with torch.no_grad():
        embeddings = hooked_model.embed(tokens).clone()
    
    # Add Gaussian noise to specified positions
    for pos in positions:
        noise = torch.randn_like(embeddings[:, pos, :]) * noise_std
        embeddings[:, pos, :] = embeddings[:, pos, :] + noise
    
    return embeddings


def run_with_noised_embeddings(hooked_model, tokens, positions, noise_std=3.0, seed=None):
    """
    Run model with noised embeddings (corrupted run).
    
    Args:
        tokens: Input tokens
        positions: Positions to add noise to
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        logits: Model output
    """
    noised_embeddings = get_noised_embeddings(hooked_model, tokens, positions, noise_std, seed)
    
    # Run model starting from embeddings instead of tokens
    # We need to manually pass through the model starting after embedding
    def embedding_hook(embeddings, hook):
        return noised_embeddings
    
    logits = hooked_model.run_with_hooks(
        tokens,
        fwd_hooks=[("hook_embed", embedding_hook)]
    )
    
    return logits


def patch_residual_stream_noise_based(hooked_model, layer, positions, clean_cache, tokens, corrupt_positions, noise_std=3.0, seed=None, patch_type='residual'):
    """
    Patch the residual stream, attention output, or MLP output at a specific layer and position(s) using noise-based corruption.
    
    Args:
        hooked_model: The hooked model to run
        layer: Layer index to patch
        positions: Position(s) to patch (restore clean activations)
        clean_cache: Cache of clean activations
        tokens: Input tokens (same as clean tokens)
        corrupt_positions: Positions where noise was added in the corrupted run
        noise_std: Standard deviation of noise used in corruption
        seed: Random seed for reproducibility
        patch_type: Type of patching - 'residual', 'attention', or 'mlp'
        
    Returns:
        patched_logits: Model output after patching
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Convert single position to list
    if isinstance(positions, int):
        positions = [positions]
    if isinstance(corrupt_positions, int):
        corrupt_positions = [corrupt_positions]
    
    if patch_type == 'residual':
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif patch_type == 'attention':
        hook_name = f"blocks.{layer}.hook_attn_out"
    elif patch_type == 'mlp':
        hook_name = f"blocks.{layer}.hook_mlp_out"
    else:
        raise ValueError(f"Invalid patch_type: {patch_type}. Must be 'residual', 'attention', or 'mlp'")
    
    clean_activation = clean_cache[hook_name]
    
    # Get noised embeddings for the corrupted run
    noised_embeddings = get_noised_embeddings(hooked_model, tokens, corrupt_positions, noise_std, seed)
    
    def embedding_hook(embeddings, hook):
        return noised_embeddings
    
    def resid_hook(act, hook):
        # Patch specified positions with clean activations
        for pos in positions:
            act[:, pos, :] = clean_activation[:, pos, :]
        return act
    
    patched_logits = hooked_model.run_with_hooks(
        tokens,
        fwd_hooks=[
            ("hook_embed", embedding_hook),
            (hook_name, resid_hook)
        ]
    )
    
    return patched_logits



import json
from typing import Dict, List, Tuple, Optional
import re

class EntityPositionParser:
    """Parse multi-hop reasoning tasks and locate entity token positions."""
    
    def __init__(self, tokenizer, hooked_model):
        self.tokenizer = tokenizer
        self.hooked_model = hooked_model
        
        # Relation to verb form mapping (from llm_knowledge_generalization.py)
        self.relation_to_verb = {
            'associated with': 'is associated with',
            'contraindication': 'is contraindicated for',
            'indication': 'is indicated for',
            'target': 'targets',
            'carrier': 'carries',
            'enzyme': 'processes',
            'expression present': 'is expressed in',
            'expression absent': 'is not expressed in',
            'interacts with': 'interacts with',
            'linked to': 'is linked to',
            'off-label use': 'is used off-label for',
            'parent-child': 'is a subtype of',
            'phenotype absent': 'lacks the phenotype',
            'phenotype present': 'has the phenotype',
            'side effect': 'causes the side effect',
            'synergistic interaction': 'synergistically interacts with',
            'transporter': 'transports',
            'ppi': 'interacts with'
        }
    
    def parse_sample(self, sample: Dict) -> Dict:
        """
        Parse a multi-hop reasoning sample to extract entity information.
        
        Args:
            sample: Dictionary containing task information with facts and generalization_task
            
        Returns:
            Dictionary with entity information and their roles
        """
        facts = sample['facts']
        gen_task = sample['generalization_task']
        
        # Extract entities from facts
        h1 = facts[0]['head']
        r1 = facts[0]['relation']
        t1 = facts[0]['tail']  # This is also h2 in a chain
        
        h2 = facts[1]['head']  # Should equal t1
        r2 = facts[1]['relation']
        t2 = facts[1]['tail']  # Final answer
        
        # Extract references from generalization task prompt
        gen_prompt = gen_task['prompt']
        
        return {
            'h1': h1,
            'r1': r1,
            't1': t1,
            'h2': h2,
            'r2': r2,
            't2': t2,
            'generalization_prompt': gen_prompt,
            'answer': gen_task['answer']
        }
    
    def _get_relation_verb_phrase(self, relation: str) -> List[str]:
        """
        Get all possible verb phrases for a relation.
        Returns list of possible phrases that might appear in text.
        add ' ' to the left of the verb to ensure correct tokenization. (for example, 'interacts with' and ' interacts with' are tokenized differently)
        """
        verb_form = self.relation_to_verb.get(relation, relation.replace('_', ' '))
        
        # Generate variations
        phrases = [' ' + verb_form]
        
        # Add gerund forms for some relations
        if 'interact' in verb_form:
            phrases.extend([' interacts with', ' interact with'])
        elif 'associate' in verb_form:
            phrases.extend([' associated with', ' associate with'])
        elif 'target' in verb_form:
            phrases.extend([' targets', ' target'])
        
        return phrases
    
    
    def find_entity_positions(self, prompt: str, entity: str, tokens_list: List[str]) -> List[int]:
        """
        Find token positions of an entity in the tokenized prompt.
        
        Args:
            prompt: The full prompt string
            entity: The entity string to find
            tokens_list: List of token strings from tokenizer
            
        Returns:
            List of token position indices
        """
        # Tokenize the entity to understand its structure
        
        # to deal with tokenizatino issues
        if prompt.find(' ' + entity) != -1:
            entity_to_tokenize = ' ' + entity
        else:
            entity_to_tokenize = entity
        entity_tokens = self.hooked_model.to_str_tokens(entity_to_tokenize)

        # Remove BOS token if present
        if entity_tokens[0] == '<|endoftext|>':
            entity_tokens = entity_tokens[1:]
        
        positions = []
        
        # Search for the entity token sequence in the full token list
        for i in range(len(tokens_list) - len(entity_tokens) + 1):
            # Check if we have a match
            match = True
            for j, entity_tok in enumerate(entity_tokens):
                # Normalize tokens for comparison (handle spaces, case, etc.)
                prompt_tok = tokens_list[i + j].strip(' ,.').lower()
                entity_tok = entity_tok.strip(' ,.').lower()
                
                if prompt_tok != entity_tok:
                    match = False
                    break
            
            if match:
                # Found a match, record all positions
                positions.extend(range(i, i + len(entity_tokens)))
                break  # Assume entity appears once
        
        return positions
    
    def find_relation_positions(self, relation: str, tokens_list: List[str]) -> List[int]:
        """
        Find token positions of relation verb phrases using the relation mapping.
        
        Args:
            relation: The relation name (e.g., 'ppi', 'associated with')
            tokens_list: List of token strings
            
        Returns:
            List of token position indices
        """
        # Get the correct verb phrases for this relation
        verb_phrases = self._get_relation_verb_phrase(relation)
        
        positions = []

        # Try each verb phrase
        for verb_phrase in verb_phrases:
            # Tokenize the verb phrase
            phrase_tokens = self.hooked_model.to_str_tokens(verb_phrase)
            if phrase_tokens[0] == '<|endoftext|>':
                phrase_tokens = phrase_tokens[1:]

            # Search for this phrase in the token list
            for i in range(len(tokens_list) - len(phrase_tokens) + 1):
                match = True
                for j, phrase_tok in enumerate(phrase_tokens):
                    prompt_tok = tokens_list[i + j].strip().lower()
                    phrase_tok = phrase_tok.strip().lower()
                    
                    if prompt_tok != phrase_tok:
                        match = False
                        break
                
                if match:
                    positions.extend(range(i, i + len(phrase_tokens)))
                    break  # Found it, no need to continue
            
            if positions:
                break  # Found with this phrase, no need to try others
        
        return positions
    
    def find_reference_positions(self, reference_pattern: str, 
                                 tokens_list: List[str]) -> List[int]:
        """
        Find token positions of reference phrases (e.g., "the protein").
        
        Args:
            reference_pattern: Pattern to search for (e.g., "protein")
            tokens_list: List of token strings
            
        Returns:
            List of token position indices
        """
        # Tokenize the reference pattern
        pattern_tokens = self.hooked_model.to_str_tokens(reference_pattern)
        if pattern_tokens[0] == '<|endoftext|>':
            pattern_tokens = pattern_tokens[1:]
        
        positions = []
        
        # Flexible matching: allow for minor variations
        for i in range(len(tokens_list) - len(pattern_tokens) + 1):
            match_score = 0
            for j, pattern_tok in enumerate(pattern_tokens):
                prompt_tok = tokens_list[i + j].strip().lower()
                pattern_tok = pattern_tok.strip().lower()
                if prompt_tok == pattern_tok or pattern_tok in prompt_tok or prompt_tok in pattern_tok:
                    match_score += 1
            
            # If most tokens match, consider it a match
            if match_score >= len(pattern_tokens) * 0.7:
                positions.extend(range(i, i + len(pattern_tokens)))
                break
        
        return positions
    
    def parse_generalization_task_positions(self, sample: Dict, 
                                           formatted_prompt: str,
                                           tokens_list: List[str]) -> Tuple[Dict[str, List[int]], Dict]:
        """
        Parse and locate all entity positions in the generalization task.
        Also marks unmatched tokens as 'others'.
        
        Args:
            sample: The task sample
            formatted_prompt: The formatted prompt string
            tokens_list: List of tokenized strings
            
        Returns:
            Tuple of (positions dictionary, entities dictionary)
        """
        entities = self.parse_sample(sample)
        
        positions = {}
        matched_positions = set()  # Track all matched positions
        
        # Find h1 positions (the starting entity)
        h1_positions = self.find_entity_positions(
            formatted_prompt, entities['h1'], tokens_list
        )
        if h1_positions:
            positions['h1'] = h1_positions
            matched_positions.update(h1_positions)
        
        # Find r1 positions using relation verb form
        r1_positions = self.find_relation_positions(entities['r1'], tokens_list)
        if r1_positions:
            positions['r1'] = r1_positions
            matched_positions.update(r1_positions)
        
        # Find t1 reference (usually "the protein" or similar)
        # This is tricky because t1 is hidden, we look for generic references
        t1_reference_patterns = [
            "the protein",
            "protein"
        ]
        for pattern in t1_reference_patterns:
            t1_positions = self.find_reference_positions(pattern, tokens_list)
            if t1_positions and not any(p in matched_positions for p in t1_positions):
                positions['t1_reference'] = t1_positions
                matched_positions.update(t1_positions)
                break
        
        # Find r2 positions using relation verb form
        r2_positions = self.find_relation_positions(entities['r2'], tokens_list)
        if r2_positions:
            positions['r2'] = r2_positions
            matched_positions.update(r2_positions)
        
        # Find t2 reference (the answer is hidden, look for "what", "which", etc.)
        t2_reference_patterns = [
            "What protein",
            "Which protein",
            "What",
            "Which"
        ]
        for pattern in t2_reference_patterns:
            t2_positions = self.find_reference_positions(pattern, tokens_list)
            if t2_positions and not any(p in matched_positions for p in t2_positions):
                positions['t2_reference'] = t2_positions
                matched_positions.update(t2_positions)
                break
        
        # Mark all unmatched positions as 'others'
        all_positions = set(range(len(tokens_list)))
        others_positions = sorted(all_positions - matched_positions)
        if others_positions:
            positions['others'] = others_positions
        
        return positions, entities


# Cross-model activation patching function
def cross_model_patch(
    source_model,
    target_model, 
    tokens,
    layer,
    positions,
    source_cache,
    patch_type='residual'
):
    """
    Patch activation from source_model into target_model
    
    Args:
        source_model: Model to get activation from
        target_model: Model to patch activation into
        tokens: Input tokens
        layer: Layer to patch
        position: Token position to patch
        source_cache: Pre-computed cache from source model
        patch_type: 'residual', 'attention', or 'mlp'
    
    Returns:
        Logits from target model with patched activation
    """
    
    # Get the activation hook name
    if patch_type == 'residual':
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif patch_type == 'attention':
        hook_name = f"blocks.{layer}.attn.hook_result"
    elif patch_type == 'mlp':
        hook_name = f"blocks.{layer}.hook_mlp_out"
    else:
        raise ValueError(f"Unknown patch_type: {patch_type}")
    
    if isinstance(positions, int):
        positions = [positions]
    # Get source activation
    source_activation = source_cache[hook_name]


    # Define patching hook
    def patch_hook(activation, hook):
        for pos in positions:
            activation[:, pos, :] = source_activation[:, pos, :]
        return activation
    
    # Run target model with patching
    with torch.no_grad():
        patched_logits = target_model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, patch_hook)]
        )
    
    return patched_logits