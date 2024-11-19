from PIL import Image
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import io
import time
from markdownify import markdownify as md
from selenium.webdriver.common.keys import Keys
from duckduckgo_search import DDGS
import os
import re
import requests

HF_TOKEN = ""

def stable_diffusion_generate_image(caption: str):
    """
    A function to create image using Stable Diffusion from caption and display it. Use specific prompt/caption to make the image accurate.

    Args:
        caption: The prompt to generate image from.
    Returns:
        A confirmation of image displayed or not
    """
    print(f"SD called with prompt: {caption}")

    def quer(payload, API_URL):
        responsesd = requests.post(API_URL, headers=headers, json=payload)
        return responsesd.content

    try:
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        image_bytes = quer({"inputs": caption, }, API_URL)
        image = Image.open(io.BytesIO(image_bytes))
        image.save(f"{caption.replace(' ','_')}.jpg")
        return {
            "status": f"{caption.replace(' ','_')}.jpg"
        }
    except Exception as error:
        return {"status": f"Error: {error}"}


def searchengineduckduckgo(keyword: str, max_results: int = 8):
    """
    A function to search the web from a keyword. ALWAYS crosschecks the information by accessing the url, the information given from search engine is not updated.

    Args:
        keyword: The keyword to search for.
        max_results: The maximum number of results to return. Increase the number for more information.
    Returns:
        A dict list of search results.
    """
    try:
        results = DDGS().text(keyword, max_results=max_results)
        return results
    except Exception as error:
        return {"status": f"Error: {error}"}


def searchenginegoogle(keyword: str):
    """
    A function to search the web from a keyword. ALWAYS crosschecks the information by accessing the url, the information given from search engine is not updated.

    Args:
        keyword: The keyword to search for.
    Returns:
        A dict list of search results.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument('window-size=1920x1080');
    # start session
    driver = webdriver.Chrome(options=options)
    returnresult = []
    try:
        # Navigate to Google
        driver.get("https://www.google.co.id")

        # # Find the search bar and input query
        search_box = driver.find_element(By.NAME, "q")
        search_query = keyword
        search_box.send_keys(search_query + Keys.RETURN)

        # Wait for results to load
        time.sleep(3)
        html = driver.page_source
        # Extract search result elements
        results = driver.find_elements(By.CSS_SELECTOR, "div.g")
        # Loop through and extract title, link, and description
        for result in results:
            try:
                description = result.text
            except Exception:
                description = "No description"
            try:
                title_element = result.find_element(By.TAG_NAME, "h3")
                title = title_element.text
            except Exception:
                title = "No title"
            try:
                link_element = result.find_element(By.CSS_SELECTOR, "a")
                link = link_element.get_attribute("href")
            except Exception:
                link = "No link"
            result_data = {
                "title": title,
                "href": link,
                "body": description
            }
            returnresult.append(result_data)
    finally:
        # Close the driver
        driver.quit()
    print(returnresult)
    return returnresult


def searchengine(keyword: str):
    """
    A function to search the web from a keyword. Always crosschecks the information by accessing the url href, the information given from search engine is not updated.

    Args:
        keyword: The keyword to search for.
    Returns:
        A dict list of search results.
    """
    print(f"Search engine called with keyword: {keyword}")
    listofresult = searchengineduckduckgo(keyword)
    # listofresult.append(searchengineduckduckgo(keyword))
    # listofresult.extend(searchenginegoogle(keyword))
    return listofresult


# def RAGPDF(filename: str, queryquestion: str):
#     """
#     A RAG function to retrieve a PDF document from a query question. This function will return a several context from pdf document according to query question.
#     Call this function if there is no information or previous information is incomplete.
#
#     Args:
#         filename: Filename of the PDF.
#         queryquestion: The question prompt.
#     Returns:
#         A list of context.
#     """
#     print(f"RAGPDF called with filename: {filename} and: {queryquestion} prompt")
#     try:
#         longtext = readPDF(filename)
#         database = vectordb(longtext, 400,350)
#         index = queryindex(database, queryquestion)
#         context = returnpassage(index, database)
#         return {"context": context}
#     except Exception as error:
#         return {"status": f"Error: {error}"}
def scrapeweb(url: str):
    """
    A function to open and scrape the web from a url.

    Args:
        url: The url to open.
    Returns:
        The text of the web page.
    """
    print(f"Scrape web called with url: {url}")
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument('window-size=1920x1080');
        # start session
        browser = webdriver.Chrome(options=options)
        browser.get(url)
        time.sleep(7)
        html = browser.page_source
        browser.quit()
        txt = md(html)
        txt = re.sub('\n+', '\n', txt)
        txt = re.sub('\n +', '\n', txt)
        txt = re.sub('\n+', '\n', txt)
        txt = re.sub('\n +', '\n', txt)
        return {'url': url, 'content': txt}
    except Exception as error:
        return {"status": f"Error: {error}"}


# def pythonrunner(code: str):
#     """
#     A function to run Python code and return its console output. Always use this to validate a generated python code.
#     Keep in mind that this function only returns the console output in text, nothing else.
#     Use display_image to display an image from a file.
#
#     Args:
#         code: The Python code to run.
#     Returns:
#         The output of the Python script.
#     """
#     print("Python runner called")
#     # Write the code to main.py
#     with open("codegenerated.py", "w") as text_file:
#         text_file.write(code)
#
#     try:
#         # Run the main.py file and capture the output
#         result = subprocess.run(
#             ["python", "codegenerated.py"],  # Command to run the Python script
#             text=True,  # Use text mode to capture output as a string
#             capture_output=True,  # Capture standard output and error
#             check=True  # Raise exception if the script fails
#         )
#         return result  # Return the standard output
#     except subprocess.CalledProcessError as e:
#         return f"Error: {e.stderr}"  # Return the error message if script fails


def pythoninterpreter(code_string: str):
    """
    A function to run Python code and return its output preview. Always use this to validate a generated python code.

    Args:
        code_string: The Python code to run.
    Returns:
        The output of the Python script.
    """
    print("Python interpreter called")

    # Step 1: Create a new notebook object with the code string
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(code_string))

    # Step 2: Execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        nb_out, _ = ep.preprocess(nb)

        # Step 3: Convert the executed notebook to HTML
        html_exporter = HTMLExporter()
        html_data, _ = html_exporter.from_notebook_node(nb_out)
        soup = BeautifulSoup(html_data, "html.parser")
        html_data = soup
        soup.find('div', class_='jp-Cell-inputWrapper').decompose()
        with open("output.html", "w") as html_file:
            html_file.write(str(html_data))
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(os.getcwd() + "/output.html")
        time.sleep(3)
        full_body_element = driver.find_element(By.TAG_NAME, "body")
        full_body_element.screenshot("output.png")
        driver.quit()
        return {'status': f"Output saved to output.png"}
    except Exception as e:
        return {'status': f"Failed to execute the notebook: {e}"}