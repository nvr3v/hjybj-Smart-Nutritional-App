---
title: AI_NutriCoach
app_file: app.py
sdk: gradio
sdk_version: 5.12.0
---
# AI NutriCoach (aka AI Dietary Crew)

AI NutriCoach is an AI-powered nutrition assistant that leverages advanced vision models and natural language processing to detect ingredients from food images, filter ingredients based on dietary restrictions, estimate calories, provide detailed nutrient analysis, and generate recipe suggestions. This project demonstrates the use of CrewAI with local models served by [LM Studio](https://lmstudio.ai) (65,536-token context window), along with other AI tools to deliver insightful and personalized nutritional feedback.

## Features

- **Ingredient Detection**  
  Detects ingredients from user-uploaded images using a vision AI model.

- **Dietary Filtering**  
  Filters detected ingredients based on user-defined dietary restrictions (e.g., vegan, gluten-free).

- **Calorie Estimation**  
  Estimates total calories from the detected ingredients.

- **Nutrient Analysis**  
  Provides a detailed breakdown of key nutrients such as protein, carbohydrates, fats, vitamins, and minerals.

- **Health Evaluation**  
  Summarizes the overall healthiness of the meal and provides a health evaluation.

- **Recipe Suggestion**  
  Generates recipe ideas based on the filtered ingredients and dietary restrictions.

## How It Works

The project is built using the CrewAI framework, which organizes agents and tasks into workflows for two primary use cases:

1. **Recipe Workflow**  
   Detects ingredients, filters them based on dietary restrictions, and suggests recipes.

2. **Analysis Workflow**  
   Directly estimates calories, performs nutrient analysis, and provides a health evaluation summary from a food image.

## Installation

### Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai) with its local server running (Developer tab > Start Server, or `lms server start`)
- A vision-capable model downloaded in LM Studio that supports a 65,536-token context. The default is `qwen/qwen3.5-9b` (`lms get qwen/qwen3.5-9b`)
- Virtual environment (optional but recommended)
- Git

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HaileyTQuach/Smart-Nutritional-App.git
   cd Smart-Nutritional-App
   ```
2. **Create and activate a virtual environment**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scriptsctivate
  ```
3. **Install the required dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
4. **(Optional) Configure LM Studio settings**: copy `.env.example` to `.env` and edit it. Without a `.env`, the app uses these defaults:

   | Variable | Default | Purpose |
   |---|---|---|
   | `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio OpenAI-compatible endpoint |
   | `LMSTUDIO_API_KEY` | `lm-studio` | Only matters if LM Studio auth is enabled |
   | `LMSTUDIO_TEXT_MODEL` | `qwen/qwen3.5-9b` | Model for the CrewAI agents and dietary filtering |
   | `LMSTUDIO_VISION_MODEL` | `qwen/qwen3.5-9b` | Model for ingredient detection and nutrient analysis (must support images) |
   | `LMSTUDIO_CONTEXT_LENGTH` | `65536` | Context window the models are loaded with |
   | `LMSTUDIO_MAX_TOKENS` | `8192` | Max tokens generated per response |
   | `LMSTUDIO_AUTO_LOAD` | `true` | Load or reload models with `LMSTUDIO_CONTEXT_LENGTH` |

## How the LM Studio connection works

- All model calls go to LM Studio's OpenAI-compatible API. The CrewAI agents use LiteLLM, and the tools use the `openai` client. No cloud API keys are needed.
- Before the first call, the app checks each model via `GET /api/v1/models`. If a model isn't loaded, or is loaded with a smaller context, the app loads it through `POST /api/v1/models/load` with `context_length=65536` and `parallel=1`, so a single request can use the whole window. Without this step, LM Studio's just-in-time loading would use the model's smaller default context.
- CrewAI is told that the window is 65,536 tokens, so it trims history only when it approaches that limit.
- Using one model for both text and vision keeps memory use down: Qwen3.5 9B at 64k context takes about 12.4 GB. If you set different text and vision models, both stay loaded at 64k, so make sure they fit in memory together.

To load the model manually instead (and set `LMSTUDIO_AUTO_LOAD=false`):

```bash
lms load qwen/qwen3.5-9b --context-length 65536 --parallel 1
```

## Usage
### Run the Application

```bash
python app.py
```

Then open http://127.0.0.1:5000, upload an image, and choose a workflow:

1. **recipe**: upload a photo of your fridge contents and optionally enter a dietary restriction (e.g. `vegan`) to get recipe ideas.
2. **analysis**: upload a photo of a finished dish to get calories, a nutrient breakdown, and a health evaluation.

## File Structure

```
Smart-Nutritional-App-Crew/
│
├── app.py                        # Gradio web app (entry point)
├── .env.example                  # LM Studio settings template
│
├── src/
│   ├── config/
│   │   ├── agents.yaml           # Configuration for agents
│   │   └── tasks.yaml            # Configuration for tasks
│   ├── crew.py                   # Crew definitions (agents, tasks, workflows)
│   ├── lmstudio.py               # LM Studio connection, model loading, 64k context
│   ├── models.py                 # Pydantic output schemas
│   └── tools.py                  # Tool definitions for ingredient detection, filtering, etc.
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or support, please contact [Hailey Thao Quach](mailto:hailey@haileyq.com).
