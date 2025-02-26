# arxiv-deep-research

This is a "Deep Research"-style agent, that attempts to generate a report answering a given research question using papers published on arxiv.org. This is achieved via API calls to the Google GenAI endpoint (Gemini or Gemma).

This is a simple hobby project, created in a single evening. It's important to note that it's not intended to compete with the sophisticated deep research agents developed by companies like Google, OpenAI, or X.ai. I'm also uncertain how closely it aligns with actual "deep research" methodologies. 

## Core functionality
This agent automates research report generation by leveraging the Google GenAI and arXiv APIs. It takes a research question as input and produces a Markdown report with cited sources.

**Process:**

1.  **Initial Search:** The agent uses an LLM (via the GenAI API) to translate the research question into multiple search queries for the arXiv API. It retrieves metadata (title, authors, abstract) for a set of potential papers.
2.  **Relevance Filtering:** The LLMs filters the retrieved papers, selecting only those deemed relevant to the research question.
3.  **Paper Acquisition and Processing:** Relevant papers are downloaded, converted from PDF to Markdown, and summarized using the LLM, focusing on key details.
4.  **Iteration and Verification:**
    * The process checks if the `--max-papers` limit is reached.
    * If not, the summaries are analyzed by the LLM to determine if sufficient information exists for report generation.
    * If more information is needed, the LLM generates new arXiv search queries, and the process repeats from step 1.
5.  **Report Generation:** Once sufficient information is gathered, the LLM generates a final Markdown report, citing the summarized papers.
## How to use

Install the requirements from `requirements.txt` and obtain a Google GenAI api key from Google AI Studio (the free version will be just fine for multiple reports per day).

Then, simply call 
```bash
export GEMINI_API_KEY="YOUR KEY" 
python arxiv_research.py --research-question "What is the best kernel size for convolutional neural networks?" --model "gemini-2.0-flash-exp" --max-papers 150
```
This will create a new research session under `./reports` where the output folder name will be the current timestamp. Once finished, the agent will output a `REPORT.md` containing the actual output and a `reading_list.json` logging it's state including all retrieved papers and their summaries.

You can also resume a research session (or simply recreate the final report, e.g., with a different model) via
```bash
python arxiv_research.py -resume <PATH_TO_SESSION>/reading_list.json
```

## Example reports

- ["How do I make very small LVLMs that generalize well?"](examples/20250225_012735//REPORT.md)
- ["What are the latest trends in adversarial robustness for image classification? Only consider papers from the last 2 years."](examples/20250224_163459/REPORT.md)
- ["signal processing flaws of convolutional neural networks"](examples/20250224_234127/REPORT.md)
- ["What is the best kernel size for convolutional neural networks?"](examples/20250226_222918/REPORT.md)

## Known issues
- Poor error handling, especially when API calls or PDF downloads fail
- Only works with Gemini
- Output path not configurable
- Gemini may hallucinate paper ID's that do not exist, which currently results in a crash due to a failures in lookups
- Citation keys may be hallucinated

## Disclaimer

Please remember, this is an AI research assistant. Double-check important information and do not trust summaries or citations blindly!