# AI Copilot

AI Interview Copilot is a desktop assistant built with Python that helps
users handle live interview scenarios. It records microphone input,
transcribes it in near real-time using Whisper (via faster-whisper), and
streams the recognized text to OpenAI GPT models to draft structured
interview responses in different styles.

## ✨ Features

-   🎤 Real-time microphone recording and transcription (Whisper).
-   🤖 AI-powered answer drafting with OpenAI GPT (streaming responses).
-   🖥️ Tkinter-based desktop UI with live transcription preview.
-   🔒 Window protection against screen sharing on Windows (via
    SetWindowDisplayAffinity).
-   📋 Copy-to-clipboard and clear text utilities.
-   ⚙️ Configurable Whisper model sizes and GPU/CPU selection.
-   📝 Detailed logging in both console and
    `logs/interview_copilot.log`.

## 📦 Requirements

-   Python 3.10+
-   sounddevice
-   numpy
-   openai
-   faster-whisper
-   python-dotenv
-   tkinter (pre-installed with most Python distributions)

## 🚀 Usage

1.  Clone the repository:
    `git clone https://github.com/SteveParadox/Ai-Copilot.git`

2.  Navigate into the project folder: `cd Ai-Copilot`

3.  Install dependencies: `pip install -r requirements.txt`

4.  Create a `.env` file in the project root with your OpenAI API key:

5.     OPENAI_API_KEY=your_api_key_here

6.  Run the application: `python ai_copilot.py`

7.  Click **Record** to start transcribing speech and generating
    interview answers.

------------------------------------------------------------------------

📝 Logs are saved in the **logs/** directory. For troubleshooting, check
**interview_copilot.log**.
