# Deploy to Hugging Face Spaces

## Step 1: Create the Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Enter a name (e.g., `sap-finance-dashboard`).
4. Select **Gradio** as the SDK.
5. Choose **Public** or **Private**.
6. Click **Create Space**.

## Step 2: Push Code
Run these commands in your terminal:

```powershell
# Add the Hugging Face remote (replace <username> with your HF username)
git remote add hf https://huggingface.co/spaces/<username>/sap-finance-dashboard

# Push the code
git push hf main
```

*Note: You will be asked for your Hugging Face username and password (token).*

## Step 3: Configure Secrets
1. Go to your Space's **Settings** tab.
2. Scroll to **Variables and secrets**.
3. Click **New secret**.
4. Name: `HUGGINGFACE_TOKEN`
5. Value: Your Hugging Face Access Token (from [Settings > Access Tokens](https://huggingface.co/settings/tokens)).

## Step 4: Enjoy!
Your app will build and start automatically. You can view the build logs in the **Logs** tab of your Space.
