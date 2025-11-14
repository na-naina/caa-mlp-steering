#!/usr/bin/env python3
"""
Test HuggingFace authentication and model access
"""

import os
import sys
from pathlib import Path

# Check for token in various locations
print("=" * 60)
print("HuggingFace Authentication Test")
print("=" * 60)
print()

# Check environment variables
print("1. Checking environment variables:")
hf_token_env = os.environ.get('HF_TOKEN', '')
hf_hub_token = os.environ.get('HUGGING_FACE_HUB_TOKEN', '')
print(f"   HF_TOKEN: {'✓ Set' if hf_token_env else '✗ Not set'}")
print(f"   HUGGING_FACE_HUB_TOKEN: {'✓ Set' if hf_hub_token else '✗ Not set'}")

# Check token file
print("\n2. Checking token file:")
token_path = Path.home() / '.cache' / 'huggingface' / 'token'
if token_path.exists():
    with open(token_path, 'r') as f:
        token_content = f.read().strip()
    print(f"   Token file exists: ✓")
    print(f"   Token length: {len(token_content)} characters")
    if len(token_content) < 10:
        print(f"   WARNING: Token seems too short!")
    else:
        print(f"   Token starts with: {token_content[:8]}...")
else:
    print(f"   Token file exists: ✗")

# Try to use huggingface_hub to login
print("\n3. Testing authentication with HuggingFace Hub:")
try:
    from huggingface_hub import HfApi, login, whoami

    # If token exists in file, use it
    if token_path.exists() and len(token_content) > 10:
        print(f"   Logging in with token from file...")
        login(token=token_content, add_to_git_credential=True)

    # Check who we are
    try:
        info = whoami()
        print(f"   ✓ Authenticated as: {info['name']}")
    except Exception as e:
        print(f"   ✗ Not authenticated: {e}")

except ImportError:
    print("   ✗ huggingface_hub not installed")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Try to access Gemma models
print("\n4. Testing model access:")
models_to_test = [
    "google/gemma-2b",  # Try the non-gated version first
    "google/gemma-2-2b",
    "google/gemma-2-9b",
]

from transformers import AutoTokenizer, AutoConfig

for model_name in models_to_test:
    try:
        print(f"\n   Testing {model_name}...")

        # Try to get config first (lighter than full model)
        config = AutoConfig.from_pretrained(model_name)
        print(f"   ✓ {model_name}: Accessible")

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "restricted" in error_msg:
            print(f"   ✗ {model_name}: Authentication required")
            print(f"      You need to accept the license at https://huggingface.co/{model_name}")
        elif "404" in error_msg:
            print(f"   ✗ {model_name}: Model not found")
        else:
            print(f"   ✗ {model_name}: {error_msg[:100]}")

print("\n" + "=" * 60)
print("INSTRUCTIONS TO FIX:")
print("=" * 60)
print("""
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with 'read' permissions
3. Save it with this command:
   echo -n "YOUR_TOKEN_HERE" > ~/.cache/huggingface/token

4. Accept model licenses at:
   - https://huggingface.co/google/gemma-2-2b
   - https://huggingface.co/google/gemma-2-9b
   - https://huggingface.co/google/gemma-2-27b

5. Test again with:
   python test_hf_auth.py
""")