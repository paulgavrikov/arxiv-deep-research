from google import genai
import json
from tqdm import tqdm
import pymupdf4llm
import os
import logging
import argparse
from datetime import datetime
from arxiv import search_arxiv, download_arxiv_pdf
from multiprocessing import Pool    


logging.basicConfig(level=logging.INFO)


MODEL_NAME = "gemini-2.0-flash-exp"

PROMPT_INITIAL = """You are a researcher assistant. Given a research topic or question, your goal is to generate search queries to find relevant papers on arxiv.org. First think step-by-step what information is needed to answer the research question. Then generate search queries to find relevant papers on arxiv.org.
Provide your answer in the following json format:
{
    "rationale": "Explain why the information is needed to answer the research question",
    "queries": ["query1", "query2", ...]
}

Your search will only return papers if the paper title or abstract matches the keywords in query. Avoid too complex queries as they will not return any results. Do not use qualifiers.

--

Research Question: %s"""

PROMPT_REFINE_QUERY = """You are a researcher assistant. You have attempted to seach for papers that answer a research question, but the search on arxiv.org did not return any relevant papers for some queries. Please refine the search queries to find relevant papers on arxiv.org. Provide your answer in the following json format:
{
    "new_queries": ["query1", "query2", ...]
}

--

Research Question: %s

Successful Queries: %s

Unsuccessful Queries: %s"""

    
PROMPT_EXTRACT_PAPER_IDS = """You are a researcher assistant. Which of the following papers from the given arxiv.org feed are relevant to the research question? 
Provide your answer in the following json format:
[
    {
        "id": "arxiv_id without the URL",
        "relevancy": "Why the paper is relevant to the research question"
    },
    ...
]

--

Research Question: %s

Feed: 
%s"""

PROMPT_SUMMARIZE_PAPER = """You are a researcher assistant. Given the following paper in Markdown format, extract the most relevant information (if any) to address the given research question. Be very detailed!

--

Research Question: %s

Paper: %s"""

PROMPT_GENERATE_REPORT = """You are a researcher assistant. Given the following summaries of papers, generate a DETAILED scientific report that answers the given research question. The report should be written in Markdown format and include the argument of the papers that are relevant to the research question. You must cite the corresponding paper for each arguments using the ID in brackets, e.g., '... [2203.08948v1]'. Generate a very detailed and long report! You must use as much paper summaries as possible to answer the research question.

--

Research Question: %s

Paper Summaries:
%s"""

PROMPT_VERIFY_RESULTS = """You are a researcher assistant. You have retrieved a set of paper summaries to answer a research question. Given the following summary list, verify that the collected information is sufficient to fully answer the research question. If the information is insufficient, generate additional search queries to find more relevant papers from arxiv.org.Provide your answer in the following json format:
{
    "information_complete": "yes" or "no",
    "rationale": "Explain why the information is sufficient or insufficient to answer the research question",
    "additional_queries": ["query1", "query2", ...]
}

The additional queries should be significantly different from the previous queries. Your search will only return papers if the paper title or abstract matches the keywords in query. Avoid too complex queries as they will not return any results. Do not use qualifiers.

--

Research Question: %s

Previous Queries: %s

Paper Summaries: 
%s"""

def extract_code(s):
    return s.replace("```json", "").replace("```markdown", "").replace("```", "").strip()


def llm_read_paper(research_question, paper_md_text):
    prompt_summarize_paper = PROMPT_SUMMARIZE_PAPER % (
        research_question,
        paper_md_text,
    )
    response = generate_response(prompt_summarize_paper)
    return extract_code(response)


def _prepare_task(paper_id):
    os.makedirs(".pdfs", exist_ok=True)
    pdf_path = os.path.join(".pdfs", f"{paper_id}.pdf")
    download_arxiv_pdf(paper_id, pdf_path, force=False)
    try:
        paper_md_text = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
    except Exception as e:
        paper_md_text = ""
        logging.error(f"Error converting {paper_id} to markdown: {e}")

    try:
        os.remove(pdf_path)
    except:
        pass
    
    return paper_id, paper_md_text


def read_papers(reading_list, data_workers=8):

    paper_ids = [id for id, paper in reading_list["papers"].items() if "summary" not in paper]

    progress_bar = tqdm(paper_ids, "Reading papers", total=len(paper_ids))

    with Pool(data_workers) as p:
        for paper_id, paper_text_md in p.imap_unordered(_prepare_task, paper_ids):
            try:
                progress_bar.set_description(f"Reading paper: {paper_id}")
                paper_summary = llm_read_paper(research_question, paper_text_md)
                reading_list["papers"][paper_id]["summary"] = paper_summary

                progress_bar.update(1)

                # update reading list    
                json.dump(
                    reading_list, open(os.path.join(session_id, "reading_list.json"), "w"), indent=2
                )     
            except Exception as e:
                logging.error(f"Error reading paper {paper_id}: {e}")
                continue

    return reading_list


def llm_extract_from_feed(research_question, result_feed):
    relevant_papers = []

    try:
        prompt_extract_paper_ids = PROMPT_EXTRACT_PAPER_IDS % (
            research_question,
            result_feed,
        )
        response = generate_response(prompt_extract_paper_ids)
        relevant_papers = json.loads(extract_code(response))

        logging.info(
            f"Found {len(relevant_papers)} relevant papers for the query: {query}"
        )
    except Exception as e:
        logging.error(f"Error extracting papers for query {query}: {e}")

    return relevant_papers


def llm_generate_queries(research_question):
    initial_prompt = PROMPT_INITIAL % research_question
    response = generate_response(initial_prompt)
    queries = json.loads(extract_code(response))["queries"]
    return queries


def llm_verify_complete(reading_list):
    # Verify results
    prompt_verify_results = PROMPT_VERIFY_RESULTS % (
        reading_list["research_question"],
        reading_list["queries"],
        reading_list["papers"],
    )

    response = generate_response(prompt_verify_results)
    verification = json.loads(extract_code(response))
    complete = verification["information_complete"] == "yes"
    if not complete:
        queries_to_run = verification["additional_queries"]
        logging.info(f"Additional queries requested: {queries_to_run}")
        logging.info(f"Explanation: {verification['rationale']}")
    else:
        logging.info("Information complete")
    return complete, queries_to_run


def llm_generate_report(reading_list):
    prompt_generate_report = PROMPT_GENERATE_REPORT % (
        reading_list["research_question"],
        reading_list["papers"],
    )
    report = generate_response(prompt_generate_report)
    return extract_code(report)


def generate_response(prompt):
    # logging.info("USER: " + prompt)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    response_text = response.text
    # logging.info("ASSISTANT: " + response_text)
    return response_text


def make_references(reading_list):
    reference_map = {}
    references = ""
    for num, (paper_id, paper) in enumerate(reading_list["papers"].items()):
        authors = ", ".join(paper['authors'])
        title = paper['title'].replace("\n", "")
        year = paper['date_published'].split('-')[0]
        references += f"[{num + 1}] {authors}, *\"{title}\"*, arXiv preprint:{paper_id}, {year}.\n\n"
        reference_map[paper_id] = num + 1
    return references, reference_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Research Assistant")
    parser.add_argument(
        "--research-question",
        type=str,
        help="Research question to answer",
        default=None,
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to a previous research session",
        default=None,
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        help="Maximum number of papers to consider",
        default=150,
    )

    # "What are the latest trends in adversarial robustness for image classification? Only consider papers from the last 2 years."

    args = parser.parse_args()

    if args.research_question is not None and args.resume is not None:
        logging.warning("Both research question and resume session provided. Ignoring research question.")

    key_store = json.load(open("keys.json"))
    GEMINI_API_KEY = key_store["GEMINI_API_KEY"]
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Extract relevant papers from search results
    research_question = args.research_question
    reading_list = {
        "research_question": research_question,
        "queries": [],
        "papers": {},
    }

    queries_to_run = []
    
    skip_queries_check = False

    if args.resume:
        with open(args.resume, "r") as f:
            session_id = args.resume.split("/")[-2]
            reading_list = json.load(f)
            research_question = reading_list["research_question"]
            logging.info(f"Resuming research session")
            skip_queries_check = True
    else:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(session_id, exist_ok=True)
        json.dump(reading_list, open(os.path.join(session_id, "reading_list.json"), "w"), indent=2)
        # Generate search queries if this is a new session
        queries_to_run = llm_generate_queries(research_question)
        logging.info(f"Generated queries: {queries_to_run}")


    while (queries_to_run is not None and len(queries_to_run) > 0) or skip_queries_check:

        skip_queries_check = False
        query_progress = tqdm(queries_to_run, "Searching papers")

        for query in query_progress:
            reading_list["queries"].append(query)  # save executed queries
            query_progress.set_description(f"Searching papers for query: {query}")
            # Search arxiv
            result_feed = search_arxiv(query, max_results=50)
            relevant_papers = llm_extract_from_feed(research_question, result_feed)

            if len(relevant_papers) == 0:
                logging.info(f"No relevant papers found for query: {query}")
                continue
                
            # add each paper
            for paper_stub in relevant_papers:
                paper_id = paper_stub["id"].split("v")[0]
                if paper_id not in reading_list["papers"].keys():
                    paper_stub["title"] = result_feed[paper_id]["title"]
                    paper_stub["authors"] = result_feed[paper_id]["authors"]
                    paper_stub["date_published"] = result_feed[paper_id]["published"]
                    paper_stub["date_updated"] = result_feed[paper_id]["updated"]
                    reading_list["papers"][paper_id] = paper_stub
        
        # Dump the data every now and then to be able to resume it
        os.makedirs(session_id, exist_ok=True)
        json.dump(reading_list, open(os.path.join(session_id, "reading_list.json"), "w"), indent=2)

        # Read each paper
        reading_list = read_papers(reading_list)

        if len(reading_list["papers"]) >= args.max_papers:
            logging.info(f"Maximum number of papers reached. Ending session.")
            break

        is_complete, queries_to_run = llm_verify_complete(reading_list)

    # Generate report
    report = llm_generate_report(reading_list)

    # add references
    references, reference_map = make_references(reading_list)

    for paper_id, citation_num in reference_map.items():
        report = report.replace(paper_id, f"{citation_num}")

    report += "\n\n### References\n"
    report += references

    logging.info("Generating report")
    with open(f"{session_id}/REPORT.md", "w") as f:
        f.write(report)
    logging.info(f"Reseach based on {len(reading_list['papers'])} papers complete.")
