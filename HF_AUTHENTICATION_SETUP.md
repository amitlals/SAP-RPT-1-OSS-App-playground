# HuggingFace Authentication Setup for SAP RPT-1-OSS

The SAP Finance Dashboard uses the **SAP-RPT-1-OSS model**, which is a gated model on Hugging Face. To enable full functionality, you need to:

1. **Accept access to the gated model**
2. **Create a Hugging Face access token**
3. **Configure the token in HF Spaces**

## Step 1: Accept Gated Model Access

1. Visit: https://huggingface.co/SAP/sap-rpt-1-oss
2. Click the **"Agree"** button to accept access to the gated model
3. You may need to be logged in to your Hugging Face account

## Step 2: Create a Hugging Face Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. **Token name**: `sap-rpt-oss-access` (or any descriptive name)
4. **Type**: Select **"Read"** (this token only needs to download models)
5. Click **"Create token"**
6. **Copy the token** (you won't be able to see it again!)

## Step 3: Add Token to HF Spaces

### Option A: Via HF Spaces Web UI (Recommended)

1. Go to your HF Space: https://huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS
2. Click **⚙ Settings** (top right)
3. Go to **"Repository secrets"** section
4. Click **"Add secret"**
5. **Name**: `HF_TOKEN`
6. **Value**: Paste your token from Step 2
7. Click **"Add secret"**
8. Wait for the Space to rebuild automatically (~2-5 minutes)

### Option B: Via Git (Alternative)

If using git to push directly, you can pass the token at build time:

```bash
# In your local repo
git push -u hf main  # Push code changes

# Then add via HF Spaces UI as above
```

## Step 4: Verify Setup

1. After the Space rebuilds, wait 1-2 minutes
2. Refresh the Space URL
3. Check the logs (scroll down on the Space page)
4. Look for: **"✓ HuggingFace authentication configured"**

If you see this message, authentication is working!

## Troubleshooting

### Error: "401 Client Error: Unauthorized"
- **Cause**: Token not set or incorrect
- **Fix**: 
  1. Verify token was accepted in HF Spaces secrets (wait for rebuild)
  2. Verify you have accepted model access (Step 1)
  3. Create a new token if needed

### Error: "HF_TOKEN not found"
- **Cause**: Environment variable not set
- **Fix**: Ensure the secret is named exactly `HF_TOKEN` (case-sensitive)

### Model still not loading
- **Cause**: Token may not have correct permissions
- **Fix**: 
  1. Delete old token
  2. Create new token with "Read" permission
  3. Accept model access again
  4. Update HF Spaces secret

## Local Development

If developing locally, set the token in your environment:

```bash
# PowerShell
$env:HF_TOKEN = "hf_..."

# Or in .env file
echo "HF_TOKEN=hf_..." >> .env
```

Then run:

```bash
python app_gradio.py
```

## Security Best Practices

- **Never commit tokens to git** (they're in .gitignore)
- **Regenerate tokens if exposed** (go to HF settings)
- **Use "Read" permission only** for model access
- **Delete unused tokens** from HF settings

## What the Dashboard Does With the Token

The token is used **only** to download the SAP-RPT-1-OSS model weights from Hugging Face Hub. The dashboard:
- Does NOT collect or store your token
- Does NOT upload any data
- Does NOT modify your Hugging Face account
- Only reads the pre-trained model file

## Additional Resources

- HF Spaces Documentation: https://huggingface.co/docs/hub/spaces
- HF Tokens Guide: https://huggingface.co/docs/hub/security-tokens
- SAP RPT-1-OSS Model: https://huggingface.co/SAP/sap-rpt-1-oss

---

**After completing these steps, your SAP Finance Dashboard will have full access to the RPT-1-OSS model for predictions and analytics!**
