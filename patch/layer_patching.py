from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch
from transformer_lens import HookedTransformer

CHAINING_TEMPLATES = {
        ('anatomy', 'expression present', 'gene/protein', 'target', 'drug'): 
            {'template': "Which drug targets the genes or proteins which are expressed in {entity_1}?",
             'relation1': "expressed in",
             'relation2': "targets",
            },
        ('anatomy', 'expression present', 'gene/protein', 'enzyme', 'drug'):
            {'template': "Which drug is catalyzed by the genes or proteins which are expressed in {entity_1}?",
                'relation1': "expressed in",
                'relation2': "catalyzed by",
            },
        ('cellular_component', 'interacts with', 'gene/protein', 'carrier', 'drug'): 
            {'template': "Which drug is carried by genes or proteins that interact with {entity_1}?",
             'relation1': "interacts with",
             'relation2': "carried by",
            },
        ('molecular_function', 'interacts with', 'gene/protein', 'target', 'drug'): 
            {'template': "Which drug targets the genes or proteins that interact with {entity_1}?",
             'relation1': "interacts with",
             'relation2': "targets",
            },
        ('effect/phenotype', 'side effect', 'drug', 'synergistic interaction', 'drug'): 
            {'template': "Which drug has a synergistica interaction with the drug that has the side effect {entity_1}?",
             'relation1': "has the side effect",
             'relation2': "has a synergistic interaction with",
            },
        ('disease', 'indication', 'drug', 'contraindication', 'disease'): 
            {'template': "Which disease is a contraindication for the drugs that is indicated for {entity_1}?",
            'relation1': "indicated for",
            'relation2': "contraindication for",
            },
        ('disease', 'parent-child', 'disease', 'phenotype present', 'effect/phenotype'): 
            {'template': "Which phenotype is present in the disease that is a sub type or super type of {entity_1}?",
            'relation1': "a sub type or super type of",
            'relation2': "present in",
            },
        ('gene/protein', 'transporter', 'drug', 'side effect', 'effect/phenotype'): 
            {'template': "Which effect is a side effect of the drug that is transported by {entity_1}?",
            'relation1': "transported by",
            'relation2': "a side effect of",
            },
        ('drug', 'transporter', 'gene/protein', 'interacts with', 'exposure'): 
            {'template': "Which exposure acts on the gene or protein that transports {entity_1}?",
            'relation1': "transports",
            'relation2': "acts on",
            },
}

MEMO_IN_CONTEXT_PROMPTS = {
    ('disease', 'associated with', 'gene/protein'):
        "cystic fibrosis->CFTR; sickle cell disease->HBB; Duchenne muscular dystrophy->DMD; disease name->",

    ('effect/phenotype', 'associated with', 'gene/protein'):
        "PTC bitter-tasting ability->TAS2R38; red-green color vision defect->OPN1LW; lactase persistence (adult lactose digestion)->LCT; phenotype name->",

    ('disease', 'contraindication', 'drug'):
        "G6PD deficiency->primaquine; asthma->propranolol; Parkinson's disease->metoclopramide; disease name->",

    ('biological_process', 'interacts with', 'exposure'):
        "DNA repair->UV radiation; inflammation->cigarette smoke; muscle hypertrophy->resistance training; biological process name->",

    ('biological_process', 'interacts with', 'gene/protein'):
        "DNA repair->BRCA1; apoptosis->CASP3; cell cycle progression->CDK1; biological process name->",

    ('cellular_component', 'interacts with', 'exposure'):
        "nucleus->ionizing radiation; mitochondrion->cyanide; endoplasmic reticulum->tunicamycin; cellular component name->",

    ('cellular_component', 'interacts with', 'gene/protein'):
        "nuclear envelope->LMNA; mitochondrion->CYCS; ribosome->RPLP0; cellular component name->",

    ('exposure', 'interacts with', 'gene/protein'):
        "UV radiation->TP53; statins->HMGCR; warfarin->VKORC1; exposure name->",

    ('exposure', 'interacts with', 'molecular_function'):
        "statins->HMG-CoA reductase activity; organophosphate pesticides->acetylcholinesterase activity; warfarin->vitamin K epoxide reductase activity; exposure name->",

    ('gene/protein', 'interacts with', 'molecular_function'):
        "DNMT1->DNA methyltransferase activity; SOD1->superoxide dismutase activity; TP53->DNA-binding transcription factor activity; gene/protein name->",

    ('gene/protein', 'interacts with', 'pathway'):
        "TP53->p53 signaling pathway; EGFR->MAPK/ERK signaling pathway; INSR->PI3K-AKT signaling pathway; gene/protein name->",
    
    ('anatomy', 'parent-child', 'anatomy'):
        "brain->cerebellum; heart->left ventricle; lung->alveolus; anatomy name->",

    ('biological_process', 'parent-child', 'biological_process'):
        "cell death->apoptosis; cell division->mitosis; signal transduction->MAPK cascade; biological process name->",

    ('cellular_component', 'parent-child', 'cellular_component'):
        "nucleus->nucleolus; mitochondrion->inner mitochondrial membrane; ribosome->large ribosomal subunit; cellular component name->",

    ('disease', 'parent-child', 'disease'):
        "cancer->breast cancer; diabetes mellitus->type 2 diabetes; infectious disease->influenza; disease name->",

    ('effect/phenotype', 'parent-child', 'effect/phenotype'):
        "pain->headache; vision loss->blindness; hearing loss->deafness; effect/phenotype name->",

    ('exposure', 'parent-child', 'exposure'):
        "tobacco smoke exposure->secondhand smoke exposure; air pollution exposure->PM2.5 exposure; radiation exposure->ionizing radiation exposure; exposure name->",

    ('molecular_function', 'parent-child', 'molecular_function'):
        "catalytic activity->kinase activity; binding->DNA binding; transporter activity->ion channel activity; molecular function name->",

    ('pathway', 'parent-child', 'pathway'):
        "metabolic pathway->glycolysis; MAPK signaling pathway->ERK signaling pathway; apoptosis pathway->intrinsic apoptosis pathway; pathway name->",
    
    ('gene/protein', 'ppi', 'gene/protein'):
        "TP53->MDM2; EGFR->GRB2; BRCA1->BARD1; gene/protein name->",

    ('drug', 'carrier', 'gene/protein'):
        "warfarin->ALB; propranolol->ORM1; levothyroxine->SERPINA7; drug name->",

    ('drug', 'enzyme', 'gene/protein'):
        "isoniazid->NAT2; fluorouracil->DPYD; irinotecan->UGT1A1; drug name->",

    ('anatomy', 'expression present', 'gene/protein'):
        "choroid->GLUL; pancreas->KCNN2; retina->RHO; anatomy name->",

    ('anatomy', 'expression absent', 'gene/protein'): "liver->RHO; retina->ALB; pancreas->TG; anatomy name->",

    ('disease', 'indication', 'drug'):
        "type 2 diabetes mellitus->metformin; hypertension->amlodipine; hypercholesterolemia->atorvastatin; disease name->",

    ('disease', 'linked to', 'exposure'):
        "lung cancer->tobacco smoking; mesothelioma->asbestos exposure; skin cancer->ultraviolet radiation; disease name->",

    ('disease', 'off-label use', 'drug'):
        "polycystic ovary syndrome->metformin; acne vulgaris->spironolactone; treatment-resistant depression->ketamine; disease name->",

    ('disease', 'phenotype absent', 'effect/phenotype'):
        "type 1 diabetes mellitus->insulin production; albinism->melanin pigmentation; alopecia areata->hair; disease name->",

    ('disease', 'phenotype present', 'effect/phenotype'):
        "influenza->fever; asthma->wheezing; cystic fibrosis->thick mucus; disease name->",

    ('drug', 'side effect', 'effect/phenotype'): 
        "diphenhydramine->drowsiness; atorvastatin->muscle pain; metformin->diarrhea; drug name->",

    ('drug', 'synergistic interaction', 'drug'): 
        "trimethoprim->sulfamethoxazole; amoxicillin->clavulanic acid; levodopa->carbidopa; drug name->",

    ('drug', 'target', 'gene/protein'): 
        "atorvastatin->HMGCR; captopril->ACE; imatinib->ABL1; drug name->",

    ('drug', 'transporter', 'gene/protein'): 
        "digoxin->ABCB1; metformin->SLC22A1; pravastatin->SLCO1B1; drug name->",
}


def find_entity_positions(
        ckpt,
        prompt: str, 
        entity: str):
    """
    Find token positions of an entity in the tokenized prompt.
    
    Args:
        prompt: The full prompt string
        entity: The entity string to find
        
    Returns:
        List of token position indices
    """
    
    # Helper function to normalize tokens for comparison
    def normalize_token(token):
        """Remove common punctuation and whitespace, convert to lowercase"""
        return token.strip(' ,.()[]{}').lower()
    
    # Try multiple tokenization strategies
    tokenization_attempts = [
        entity,           # Original entity
        ' ' + entity,     # With leading space
    ]
    
    tokens_list = ckpt.to_str_tokens(prompt, prepend_bos=False)
    
    for entity_to_tokenize in tokenization_attempts:
        entity_tokens = ckpt.to_str_tokens(entity_to_tokenize, prepend_bos=False)
        
        # Remove BOS token if present
        if entity_tokens and entity_tokens[0] == ckpt.tokenizer.bos_token:
            entity_tokens = entity_tokens[1:]
        
        if not entity_tokens:
            continue
            
        positions = []
        
        # Search for the entity token sequence in the full token list
        for i in range(len(tokens_list) - len(entity_tokens) + 1):
            # Check if we have a match
            match = True
            for j, entity_tok in enumerate(entity_tokens):
                # Normalize tokens for comparison
                prompt_tok_norm = normalize_token(tokens_list[i + j])
                entity_tok_norm = normalize_token(entity_tok)
                
                if prompt_tok_norm != entity_tok_norm:
                    match = False
                    break
            
            if match:
                # Found a match, record all positions
                positions.extend(range(i, i + len(entity_tokens)))
                return positions  # Return immediately on first successful match
    
    # If exact matching fails, try fuzzy matching based on original string
    # This handles cases where tokenization is very different
    prompt_lower = prompt.lower()
    entity_lower = entity.lower()
    
    if entity_lower in prompt_lower:
        # Find the character position
        char_start = prompt_lower.find(entity_lower)
        char_end = char_start + len(entity_lower)
        
        # Map character positions to token positions
        positions = []
        current_char_pos = 0
        
        for token_idx, token in enumerate(tokens_list):
            token_start = current_char_pos
            token_end = current_char_pos + len(token)
            
            # Check if this token overlaps with the entity span
            if not (token_end <= char_start or token_start >= char_end):
                positions.append(token_idx)
            
            current_char_pos = token_end
        
        if positions:
            return positions
    
    return []


def cross_layer_self_patch(
    source_cache,
    target_model, 
    tokens,
    source_layer_id,
    target_layer_id,
    positions,
    patch_type='residual',
    source_positions=None
):
    """
    Patch activation from source_model into target_model
    
    Args:
        target_model: Model to patch activation into
        tokens: Input tokens
        source_layer_id: Layer ID(s) to get activation from (int or list of ints)
        target_layer_id: Layer ID(s) to patch activation into (int or list of ints)
        positions: Token position(s) to patch
        source_cache: Pre-computed cache from source model
        patch_type: 'residual', 'attention', or 'mlp'
        source_positions: in case we want to use the activation from a different prompt
    
    Returns:
        Logits from target model with patched activation
    """
    
    # Convert single values to lists for uniform handling
    if isinstance(source_layer_id, int):
        source_layer_id = [source_layer_id]
    if isinstance(target_layer_id, int):
        target_layer_id = [target_layer_id]
    if isinstance(positions, int):
        positions = [positions]
    
    # Validate that source and target layer lists have the same length
    assert len(source_layer_id) == len(target_layer_id), \
        f"source_layer_id and target_layer_id must have the same length, got {len(source_layer_id)} and {len(target_layer_id)}"
    
    # Prepare all hook pairs
    fwd_hooks = []
    
    for src_layer, tgt_layer in zip(source_layer_id, target_layer_id):
        # Get the activation hook name for target layer
        if patch_type == 'residual':
            hook_name = "hook_resid_post"
        elif patch_type == 'residual_pre':
            hook_name = "hook_resid_pre"
        elif patch_type == 'attention':
            hook_name = "hook_attn_out"
        elif patch_type == 'mlp':
            hook_name = "hook_mlp_out"
        else:
            raise ValueError(f"Unknown patch_type: {patch_type}")
        
        # Get source activation from source layer
        source_activation = source_cache[f'blocks.{src_layer}.{hook_name}'].to(target_model.cfg.device)
        
        
        if source_positions is None:
            source_positions = positions # same prompt by default, so patch the same positions
        # Define patching hook (use closure to capture source_activation)
        def make_patch_hook(src_act):
            def patch_hook(activation, hook):
                for src_pos, pos in zip(source_positions, positions):
                    activation[:, pos, :] = src_act[:, src_pos, :]
                return activation
            return patch_hook
        
        fwd_hooks.append((f'blocks.{tgt_layer}.{hook_name}', make_patch_hook(source_activation)))
    
    # Run target model with all patching hooks
    with torch.no_grad():
        patched_logits = target_model.run_with_hooks(
            tokens,
            fwd_hooks=fwd_hooks
        )
    
    return patched_logits


# Helper function to load a model
def load_checkpoint(checkpoint_path, 
                    base_model_name,
                    device='cuda:0',
                    tl_model_name=None):
    """Load a LoRA checkpoint and merge it with base model.
    
    Args:
        tl_model_name: Override model name for transformer_lens config lookup.
                       Use when the actual model isn't in TL's registry but shares
                       architecture with a registered model (e.g. Phi-3.5 → Phi-3).
    """
    
    # Load base model
    base_model_temp = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu",
    )
    
    if checkpoint_path is None or checkpoint_path == "base":
        merged_model = base_model_temp
    else:
        # Load and merge LoRA
        peft_model = PeftModel.from_pretrained(base_model_temp, checkpoint_path)
        merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    merged_model.to(device)
    
    # Create HookedTransformer
    hooked = HookedTransformer.from_pretrained_no_processing(
        tl_model_name or base_model_name, 
        hf_model=merged_model, 
        dtype=torch.float32,
        device=device
    )

    return hooked

def cross_layer_self_patch_generate(
    source_cache,
    target_model, 
    prompt_tokens,
    source_layer_id,
    target_layer_id,
    positions,
    patch_type='residual',
    source_positions=None,
    max_new_tokens=64,
):
    """
    Patch activation from source_cache into target_model and generate tokens with greedy sampling.
    
    Args:
        source_cache: Pre-computed cache from source model
        target_model: Model to patch activation into
        prompt_tokens: Input prompt tokens (not including answer)
        source_layer_id: Layer ID(s) to get activation from (int or list of ints)
        target_layer_id: Layer ID(s) to patch activation into (int or list of ints)
        positions: Token position(s) to patch in the prompt
        patch_type: 'residual', 'attention', or 'mlp'
        source_positions: in case we want to use the activation from a different prompt
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        Generated token IDs (including prompt tokens)
    """
    import torch
    
    # Convert single values to lists for uniform handling
    if isinstance(source_layer_id, int):
        source_layer_id = [source_layer_id]
    if isinstance(target_layer_id, int):
        target_layer_id = [target_layer_id]
    if isinstance(positions, int):
        positions = [positions]
    
    if source_positions is None:
        source_positions = positions
    
    assert len(source_layer_id) == len(target_layer_id), \
        f"source_layer_id and target_layer_id must have the same length"
    
    # Get hook name based on patch type
    if patch_type == 'residual':
        hook_name = "hook_resid_post"
    elif patch_type == 'residual_pre':
        hook_name = "hook_resid_pre"
    elif patch_type == 'attention':
        hook_name = "hook_attn_out"
    elif patch_type == 'mlp':
        hook_name = "hook_mlp_out"
    else:
        raise ValueError(f"Unknown patch_type: {patch_type}")
    
    # Initialize generated tokens with prompt
    generated_tokens = prompt_tokens.clone()
    prompt_len = prompt_tokens.shape[1]
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            current_len = generated_tokens.shape[1]
            
            # Build patching hooks - only patch positions that exist in current sequence
            fwd_hooks = []
            for src_layer, tgt_layer in zip(source_layer_id, target_layer_id):
                source_activation = source_cache[f'blocks.{src_layer}.{hook_name}'].to(target_model.cfg.device)
                
                # Filter positions that are within current sequence length
                valid_pairs = [(src_pos, pos) for src_pos, pos in zip(source_positions, positions) 
                               if pos < current_len]
                
                if valid_pairs:
                    def make_patch_hook(src_act, pairs):
                        def patch_hook(activation, hook):
                            for src_pos, pos in pairs:
                                activation[:, pos, :] = src_act[:, src_pos, :]
                            return activation
                        return patch_hook
                    
                    fwd_hooks.append((f'blocks.{tgt_layer}.{hook_name}', make_patch_hook(source_activation, valid_pairs)))
            
            # Run model with hooks
            if fwd_hooks:
                logits = target_model.run_with_hooks(generated_tokens, fwd_hooks=fwd_hooks)
            else:
                logits = target_model(generated_tokens)
            
            # Greedy sampling: get the token with highest probability at last position
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append the new token
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == target_model.tokenizer.eos_token_id:
                break
    
    return generated_tokens