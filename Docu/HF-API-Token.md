# Step-by-Step Guide to Obtain a Hugging Face API Token

Log in to Your Hugging Face Account:

Visit the Hugging Face website: Hugging Face
Log in using your credentials.
Go to Your Settings:

Click on your profile picture in the top right corner of the website.
Select "Settings" from the dropdown menu.
Navigate to Tokens:

In the settings, find the section called "Access Tokens" or "API Tokens".
Click on it to open the token management page.
Create a New Token:

Click the button "New token" or "Create new token".
Give the token a name, e.g., "API Token for my project".
Select the desired permissions (usually "Read" and "Write" are sufficient for most applications).
Save the Token:

Click "Create" or "Generate".
Copy the generated token and store it securely, as it will only be shown once.
Using the Token in Your Script
Once you have your token, you can use it in your script as follows:

Set the Token as an Environment Variable:

Set the token as an environment variable to use it in your script.
In the command line (Linux/MacOS):

bash
Code kopieren
export HUGGINGFACE_HUB_TOKEN="your_huggingface_token"
In the command line (Windows):

cmd
Code kopieren
set HUGGINGFACE_HUB_TOKEN=your_huggingface_token

By following these steps, you ensure that your Hugging Face API token is valid and correctly used in your script. If you need further assistance, feel free to ask!
