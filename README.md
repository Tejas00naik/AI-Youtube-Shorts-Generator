# AI YouTube Shorts Generator

AI YouTube Shorts Generator is a powerful Python tool that automatically creates engaging YouTube Shorts from long-form videos. It leverages advanced AI technologies including Whisper for transcription and GPT-4 for content analysis and narrative planning.

[![YouTube Tutorial](https://img.shields.io/badge/YouTube-Tutorial-red)](https://youtu.be/dKMueTMW1Nw)
[![Medium Article](https://img.shields.io/badge/Medium-Article-black)](https://medium.com/@anilmatcha/ai-youtube-shorts-generator-in-python-a-complete-tutorial-c3df6523b362)

![Demo](https://github.com/user-attachments/assets/3f5d1abf-bf3b-475f-8abf-5e253003453a)

## ✨ Features

- **Smart Content Analysis**: Automatically identifies key moments and highlights
- **Multiple Narrative Modes**: Supports different content styles (tutorials, highlights, stories, etc.)
- **AI-Powered Scripting**: Generates natural-sounding voiceovers and captions
- **Automated Editing**: Handles video cropping, transitions, and effects
- **Speaker Detection**: Identifies and tracks speakers in the video
- **Vertical Format**: Optimized output for YouTube Shorts (9:16 aspect ratio)
- **Customizable Output**: Adjust tone, style, and duration to match your brand

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg
- OpenCV

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator.git
   cd AI-Youtube-shorts-generator
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## 🎯 Usage

### Basic Usage

```bash
python main.py
```

When prompted, enter the YouTube URL of the video you want to convert to a Short.

### Advanced Usage

Use the Python API for more control:

```python
from core.input_parser import parse_user_input
from llm.narrative_planner import generate_narrative_plan

# Parse user input
user_input = "Create a 45-second tutorial about Python decorators for beginners"
params = parse_user_input(user_input)

# Generate narrative plan
transcript = "Your video transcript goes here..."
plan = generate_narrative_plan(
    transcript=transcript,
    mode=params['mode'],
    tone=params['tone'],
    **params['params']
)
```

## 🏗️ Project Structure

```
ai-youtube-shorts-generator/
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── input_parser.py      # Parses user input and extracts parameters
│   └── validator.py         # Validates generated content
├── llm/                     # LLM integration
│   ├── __init__.py
│   └── narrative_planner.py # Generates narrative plans
├── media/                   # Media processing
│   ├── audio.py
│   ├── video.py
│   └── effects.py
├── tests/                   # Test suite
├── .env.example            # Example environment variables
├── requirements-dev.txt     # Development dependencies
└── README.md               # This file
```

## ⚙️ Configuration

Edit the `.env` file to configure the application:

```ini
# OpenAI API (required for narrative generation)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4  # or another supported model

# Video Settings
DEFAULT_VIDEO_WIDTH=1080
DEFAULT_VIDEO_HEIGHT=1920  # Vertical format for Shorts
MAX_VIDEO_DURATION=58  # Maximum duration in seconds

# Audio Settings
SAMPLE_RATE=44100
BITRATE='192k'
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Generate a coverage report:

```bash
pytest --cov=./ --cov-report=html
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is a v0.1 release and might have some bugs. Please report any issues on the [GitHub Repository](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator).

## 🌟 Related Projects

- [AI Influencer Generator](https://github.com/SamurAIGPT/AI-Influencer-Generator)
- [Text to Video AI](https://github.com/SamurAIGPT/Text-To-Video-AI)
- [Faceless Video Generator](https://github.com/SamurAIGPT/Faceless-Video-Generator)
- [AI B-roll Generator](https://github.com/Anil-matcha/AI-B-roll)
- [No-code AI YouTube Shorts Generator](https://www.vadoo.tv/clip-youtube-video)
- [Sora AI Video Generator](https://www.vadoo.tv/sora-ai-video-generator)
