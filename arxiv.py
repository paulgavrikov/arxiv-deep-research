import requests
import urllib
from xml.dom import minidom
import urllib.request
import os


def download_arxiv_pdf(id, pdf_path, force=False):
    pdf_url = f"https://arxiv.org/pdf/{id}"

    if os.path.exists(pdf_path) and not force:
        return

    with open(pdf_path, "wb") as f:
        f.write(requests.get(pdf_url).content)


def parse_arxiv_feed(xml_feed):
    xml_doc = minidom.parseString(xml_feed)
    feed = xml_doc.getElementsByTagName('feed')[0]
    entries = feed.getElementsByTagName('entry')

    feed_entries = {}

    for entry in entries:
        paper_id = entry.getElementsByTagName('id')[0].firstChild.nodeValue.replace("http://arxiv.org/abs/", "")
        title = entry.getElementsByTagName('title')[0].firstChild.nodeValue
        published = entry.getElementsByTagName('published')[0].firstChild.nodeValue
        updated = entry.getElementsByTagName('updated')[0].firstChild.nodeValue
        summary = entry.getElementsByTagName('summary')[0].firstChild.nodeValue.strip()
        
        authors = [entry.getElementsByTagName('name')[0].firstChild.nodeValue for entry in entry.getElementsByTagName('author')]
        categories = [entry.getAttribute('term') for entry in entry.getElementsByTagName('category')]

        entry_dict = {
            "title": title,
            "authors": authors,
            "published": published,
            "updated": updated,
            "categories": categories,
            "summary": summary,
        }
        
        feed_entries[paper_id] = entry_dict
    
    return feed_entries


def search_arxiv(query, max_results=50):
    enc_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{enc_query}&max_results={max_results}&sortBy=relevance"
    data = urllib.request.urlopen(url)
    xml_feed = data.read().decode("utf-8")
    feed_entries = parse_arxiv_feed(xml_feed)
    return feed_entries
